# DeepSeek V4\.1 MLA

# prefill

```Python
└── _forward_prefill
    ├── 1. 获取寻址索引 (Indices Acquisition)
    │   ├── SWA-only 层: top_k = 0 (强制屏蔽远期)
    │   ├── CSA (C4A) 层: 从打分网络 (Indexer) 提取局部 Top-K 索引
    │   └── HCA (C128A) 层: 读取元数据中预先 `arange` 好的全量连续索引
    │
    ├── 2. 动态显存规划 (Dynamic Chunk Planning)
    │   └── get_prefill_chunk_plan
    │       ├── 计算压缩区局部最大宽度 (chunk_N)
    │       ├── 计算 Chunk 整体最大宽度 (chunk_M = N + SWA)
    │       └── 划定防 OOM 的安全微批次边界 (chunk_start -> chunk_end)
    │
    └── 3. 微批次流水线执行 (Chunk-by-chunk Execution)
        └── for chunk in chunk_plan:
            ├── 申请内存: 向 Workspace 索要一块绝对连续的 1D/2D 显存 (kv)
            ├── 搬运前段: dequantize_and_gather (将压缩历史反量化，塞入 offset=0)
            ├── 搬运后段: dequantize_and_gather (将SWA近期数据反量化，塞入 offset=chunk_N)
            ├── 拼装指针: combine_topk_swa_indices (将逻辑索引映射为带偏置的全局一维物理坐标)
            │
            └── [底层算子调用] flash_mla_sparse_fwd
```



prefill接口如下：

```YAML
def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
```

其中DeepseekV4FlashMLAMetadata和DeepseekSparseSWAMetadata分别为：

```Python
@dataclass
class DeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int

    # Pre-computed C128A metadata (compress_ratio == 128 only).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    c128a_prefill_topk_indices: torch.Tensor | None = None

```

```Python
@dataclass
class DeepseekSparseSWAMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    query_start_loc_cpu: torch.Tensor | None = None  # [num_seqs + 1]

    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, window_size]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Pre-computed prefill metadata shared across all DeepseekV4 attention layers.
    prefill_seq_lens: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_window_size: int = 0
    prefill_max_model_len: int = 0
    prefill_max_num_batched_tokens: int = 0

    # Per-layer-type FlashMLA tile-scheduler metadata. One FlashMLASchedMeta
    # per present DeepseekV4 layer type, shared across all ~60 layers of that type
    # within a decode step. The first forward call of a given type triggers
    # the in-kernel planner (which also allocates tile_scheduler_metadata and
    # num_splits via PyTorch's graph-aware allocator); subsequent same-type
    # calls skip planning and reuse the plan. Fresh instance per build(), so
    # have_initialized is always False at the start of a step and the plan
    # is re-derived from current seq_lens / topk_length on replay.
    # None for layer types the model does not use (or when num_decode_tokens
    # is zero).
    tile_sched_swaonly: "FlashMLASchedMeta | None" = None
    tile_sched_c4a: "FlashMLASchedMeta | None" = None
    tile_sched_c128a: "FlashMLASchedMeta | None" = None

    def get_prefill_chunk_plan(
        self, compress_ratio: int, prefill_chunk_size: int
    ) -> list[tuple[int, int, int, int]]:
        if self.num_prefills == 0:
            return []

        assert self.prefill_seq_lens_cpu is not None
        assert self.prefill_query_lens_cpu is not None

        # query_len <= max_num_batched_tokens and
        # gather_len = query_len + min(prefix_len, window_size - 1), so the
        # worst-case gathered width is bounded by
        # max_num_batched_tokens + window_size - 1. The compressed prefix pool
        # is bounded by ceil(max_model_len / compress_ratio).
        max_workspace_area = prefill_chunk_size * (
            (
                0
                if compress_ratio <= 1
                else cdiv(self.prefill_max_model_len, compress_ratio)
            )
            + self.prefill_window_size
            + self.prefill_max_num_batched_tokens
        )
        prefix_lens_cpu = self.prefill_seq_lens_cpu - self.prefill_query_lens_cpu
        gather_lens_cpu = self.prefill_query_lens_cpu + torch.clamp(
            prefix_lens_cpu, min=0, max=self.prefill_window_size - 1
        )
        compressed_lens_cpu = (
            torch.zeros_like(self.prefill_seq_lens_cpu)
            if compress_ratio <= 1
            else torch.div(
                self.prefill_seq_lens_cpu,
                compress_ratio,
                rounding_mode="floor",
            )
        )

        chunk_plan: list[tuple[int, int, int, int]] = []
        chunk_start = 0
        while chunk_start < self.num_prefills:
            chunk_max_compressed = int(compressed_lens_cpu[chunk_start].item())
            chunk_max_gather = int(gather_lens_cpu[chunk_start].item())
            chunk_end = chunk_start + 1

            while chunk_end < self.num_prefills:
                candidate_max_compressed = max(
                    chunk_max_compressed,
                    int(compressed_lens_cpu[chunk_end].item()),
                )
                candidate_max_gather = max(
                    chunk_max_gather,
                    int(gather_lens_cpu[chunk_end].item()),
                )
                candidate_width = candidate_max_compressed + candidate_max_gather
                candidate_area = (chunk_end - chunk_start + 1) * candidate_width
                if candidate_area > max_workspace_area:
                    break
                chunk_max_compressed = candidate_max_compressed
                chunk_max_gather = candidate_max_gather
                chunk_end += 1

            chunk_plan.append(
                (
                    chunk_start,
                    chunk_end,
                    chunk_max_compressed,
                    chunk_max_compressed + chunk_max_gather,
                )
            )
            chunk_start = chunk_end

        return chunk_plan
```

可以看到，DeepseekV4FlashMLAMetadata和DeepseekSparseSWAMetadata分别包含着block\_table和block\_size字段，也就是说，本质上在\_forward\_prefill中，是替代了一部分原本在kernel内进行的操作，包括sparse访存等。

# decode

decode的反量化发生在flash\_mla\_with\_kvcache内，基本分支派发如下：

```Python
_forward_decode
    ├── 1. 逻辑地址向物理地址翻译 (Address Translation)
    │   ├── SWA-only 层: topk_indices = None
    │   ├── CSA (C4A) 层: compute_global_topk_indices_and_lens
    │   │   └── 拿着打分网络给的“逻辑块号”，查 block_table，翻译成“全局物理页表地址”
    │   └── HCA (C128A) 层: 直接提取预计算的“全局物理页表地址”
    │
    ├── 2. CUDA Graph 状态锁定 (Tile Scheduler)
    │   └── 根据层类型 (swaonly/c4a/c128a) 分配特定的 metadata，确保显存复用
    │
    ├── 3. 构建主副“双轨”参数 (Dual-Track Assembly)
    │   ├── 主轨 (近期): swa_cache (块数据), swa_indices (块指针)
    │   └── 副轨 (远期): kv_cache (压缩块数据), topk_indices (翻译好的物理指针)
    │
    └── [底层算子调用] flash_mla_with_kvcache
        └── 双指针寻址: 不做任何 Gather 搬运！GPU 直接拿着主副两套指针，
            去碎片化的 Paged Cache 里跳跃捞取 FP8 数据，瞬间反量化并相加解码！
```

## decode的反量化

首先，decode模式下的kcache需要进行反量化，kcache在显存中以特定形式存储：

```Python
"""
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.
    """
```

在抓取某个token的kcache时，首先要计算block\_idx和token在block内的索引rel\_idx\_in\_block ：

```C++
int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);   // Use uint32_t division and mod to improve performance
int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size < page_block_size, so there will be no illegal-memory-access error
```

然后分别定位到某个数据部分和量化因子部分：

```Python
gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
```

先加载量化因子部分，将e8m0的量化因子转换为bf16：

```Python
fp8_e8m0 scales_e8m0[NUM_SCALES];
*(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
CUTE_UNROLL
for (int i = 0; i < NUM_SCALES; i += 2) {
    *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
}
```

然后是加载数据的nope部分，这部分为e4m3，升精度到fp32后乘bf16的量化因子，最后再使用rn舍入转回bf16：

```C++
fp8* gK_nope = gK_base + (lane_idx/8)*16;
    if (token_index == -1) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; ++i)
            scales[i] = (bf16)0.0f;
    }
    CUTE_UNROLL
    for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE/64; dim_idx += 1) {
        fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim_idx*64);   // We use EVICT_LAST here since gK_base may not be aligned to 32B (for V3.2) and the performance is the best among all cache hints (for MODEL1)
        bf16 scale = scales[MODEL_TYPE == ModelType::V32 ? dim_idx/2 : dim_idx];
        auto dequant_and_save_bf16x8 = [&](const fp8x8 &data, int offset) {
            int smem_offset = (dim_idx*64 + offset) * TOPK_BLOCK_SIZE;
            bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
            *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
            if constexpr (CLUSTER_SIZE == 2) {
                st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
            }
        };
        if (token_index == -1)
            *(uint128_t*)(&cur_fp8x16) = uint128_t();
        dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
        dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
    }
```

最后把rope部分拼接回来：

```C++
bf16* gK_rope;
    if constexpr (MODEL_TYPE == ModelType::V32) {
        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE+NUM_SCALES*sizeof(float)) + (lane_idx/8)*8;
    } else {
        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE) + (lane_idx/8)*8;
    }
    bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
    bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

    CUTE_UNROLL
    for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
        bf16x8 cur_bf16x8 = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx*32);
        if constexpr (MODEL_TYPE == ModelType::V32) {
            // NOTE We do not need to mask the RoPE part for V3.2 since it isn't involved in the SV gemm
        } else {
            if (token_index == -1)
                *(uint128_t*)(&cur_bf16x8) = uint128_t();
        }
        int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
        *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
        if constexpr (CLUSTER_SIZE == 2) {
            st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
        }
    }
```

## decode时的显存排布

有这样一段注释：

```Python
"""
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.
    """
```

首先计算每个token所需的实际空间，bytes\_per\_token在deepseekv4下等于448\*sizeof\(e4m3\)\+64\*sizeof\(bf16\)\+\(7\+1\)\*sizeof\(e8m0\)=584。

