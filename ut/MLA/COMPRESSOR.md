# DSV4.1 Compressor（KV/score 压缩器）— 调研与分类

> 目标硬件：**NVIDIA H200 / sm90（Hopper，compute capability 9.0）**
> 代码基线：`E:\dsv41\vllm_0.32.0\vllm-v0.32.0rc0\vllm-v0.32.0rc0`（vLLM v0.32.0rc0），下文所有 `文件:行号` 都相对这个目录
> 模型基线：DeepSeek-V4.1，40 层 + 3 层 MTP（`E:\dsv41\dsv41_part`，只有 config 和 index，**没有 `.safetensors` 分片**，所以张量形状与 dtype 来自 config 加参考实现，不是读张量得到的）
> 生成方式：`compressor\build_doc.py`（**请勿手工编辑**；生成时会先校验每个行号锚点与每个推算数字）

---

## 0. 结论速览

1. **compressor 只存在于 layer 2 / 8 / 14 / 20**（`kv_source_layer_ids`），也就是 4 个「kv SOURCE」层；其余 38 层复用别人的压缩结果，自己没有 compressor。
2. 它的作用是把**每个 ratio 个 token 压成 1 个 512 维的 latent**：先按「逐维 softmax」做软门控池化，再 RMSNorm，落到 **BF16**。
3. **H200 上真正生效的只有一条链**：FlashMLA 后端 → `use_fp8_ds_mla_layout=True` → 每次写一条 **584 B** 记录（448 B NoPE 用 fp8 + 64 维 RoPE 用 bf16 + 7 个 UE8M0 缩放 + 1 字节填充）。528 B（MXFP8）和 288 B（NVFP4）两种记录在 sm90 上**都到不了**。
4. **latent 是「pre-RoPE」的，而且下游只读**：主压缩 cache 和 indexer K cache 两条路都从它派生，RoPE 各自加，所以它们可以并行。
5. 一共涉及**三个 cache 加一个 ring**：主压缩 KV（584 B/记录，只有 kv-source 有）、SWA cache（每层都有，H200 上**也恰好**是 584 B/记录）、indexer K cache（132 B/行）、`state_cache` FP32 ring（4096 B/行，只有 ratio 2 有）。

---

## 1. 术语与数学定义

| 记号 | 含义 |
|---|---|
| `T` | 本步的 token 总数（所有请求加起来） |
| `ratio` | 本层 `compress_ratios[layer]`，**compressor 只支持 1 或 2** |
| 组（group） | 连续的 `ratio` 个 token；latent 只在**组边界**（组的最后一个 token）产生 |
| `hidden` | `hidden_states`，`[T, 5120]` bf16 |
| `kv_score` | compressor 的输入，`[T, 512*ratio]` FP32。**前 512 列是 kv，后 512 列是门控分数** |
| `latent` | compressor 的输出，`[T, 512]` **BF16**，pre-RoPE |

**池化公式（关键，不是 sigmoid，也不是每 token 一个标量）**：

```
给定组内 ratio 个 token，逐维 d ∈ [0,512) 独立做：
    pooled[d] = Σ_i  kv_i[d] · softmax_i(score_i[d])
```

也就是说 **softmax 是沿「组内 token」这一维、对每一维 d 单独做的**，所以每个维度有自己的一组权重。源码用一行表达（`fused_compress_quant_cache.py:164` 和 `:211`）：

```python
tl.sum(kv * tl.softmax(score, 0), 0)   # axis 0 = 组内 token
```

`score` 只用于算权重，**不会进入输出**。

---

## 2. H200/sm90 运行时后端门控矩阵（关键）

光看 `compress_ratio` 不够：**哪些 kernel 真正跑得到，取决于后端和 dtype 门控**。下表是在 H200 上逐条核实的真值（编号沿用调研阶段的完整矩阵，只列出**与 compressor 有关**的那些）：

| # | 门控 | H200 真值 | 代码位置 | 影响 |
|---|---|---|---|---|
| G1 | `_use_v41_mxfp8_kv_record()` | **False** | `platforms/interface.py:588` | 528 B MXFP8 记录不可达 |
| G2 | `dsa_indexer_uses_fp4` | **False** | `mla/indexer.py:59-96` | indexer K 用 FP8，不是 MXFP4 |
| G3 | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 可用 | **True** | CUDA 自定义 op | SWA cache 走融合算子 |
| G6 | `_ON_GFX950` | **False** | 定义 `fused_compress_quant_cache.py:11-14`（`is_rocm()` 分支）；消费点 `:273, 337, 403` | `max_encoded = 255.0`（不是 254） |
| G7 | `is_device_capability_family(90)` | **True** | `sparse_mla.py:88-90` | kernel block size = **64** |
| G10 | `major == 12` 之类的判断 | **False** | `nvidia/model.py:221, 233` | 不选 `DeepseekV4FlashInferSM120Attention` |
| G12 | `dsa_indexer_uses_fp4(vllm_config)` | **False** | `vllm/v1/attention/backends/mla/indexer.py` | indexer K cache = **132 B/行**（FP8），不是 68 B（MXFP4） |
| G13 | `use_fp8_ds_mla_layout`（类属性） | **True** | `attention.py:196`；`flashmla.py:50` 继承 | 主压缩 cache 用 packed uint8 布局 |
| G14 | `current_platform.is_rocm() and block_size > 1` | **False** | `indexer_k_store.py:80` | indexer K cache 不按 ROCm 的 "SHUFFLE" tile 写 |

**H200 实际走哪条**：唯一生效的是一条链 —— `DeepseekV4FlashMLAAttention`（`nvidia/model.py:226-231, 243`；`nvidia/flashmla.py:50`）→ `use_fp8_ds_mla_layout=True`（`attention.py:196`）→ packed `uint8`、**记录 584 B**、**页对齐 576 B**（`:513, 516-518, 522`）→ 解 `_rope_quant_insert_kernel`（`fused_compress_quant_cache.py:255-259`）；kernel block size = **64**（`sparse_mla.py:90`），`JOIN_ROW_PTRS=True`（`:19`）、PDL 关闭（`:113, 253`）、`SANITIZE_CACHE_NANS=False`（`:273`）。

**到不了的分支（H200 上不可达）**：

| 分支 | 为什么到不了 |
|---|---|
| 528 B MXFP8 记录 | 需 `kv_mxfp8=True`，即 G1 为真（SM100）（`:254-259, 354-405`） |
| 288 B NVFP4 记录 | 需 `kv_cache_dtype="nvfp4_ds_mla"`，sm90 启动即 `ValueError`（`attention.py:523-527`）；只有 `FlashMLAMegaAttnBackend` 声明该 dtype，而它 `major == 10`（`sparse_mla.py:235, 263`） |
| 普通行（bf16 / per-tensor fp8） | 需 `use_fp8_ds_mla_layout=False`。⚠️ **不能一刀切说「只有 CUDA FlashInfer 类这么设」**：同族的 `DeepseekV4FlashInferSM120Attention`（`flashinfer_sparse.py:602`）在 **`:607` 设的是 `True`**；只有 `DeepseekV4FlashInferMLAAttention`（`:246`）才是 `False`（`:251`），而它的 `supports_compute_capability` 只接受 major ∈ {10, 12}（`:137-138`），sm90 不满足 |
| ROCm 专属分支（`_JOIN_ROW_PTRS=False` 手写稳定 softmax） | ROCm 专属（`fused_compress_quant_cache.py:16-19`） |
| ratio 4 / 128 | 构造时抛 `NotImplementedError`（`compressor.py:196-201`） |
| V4.0 的 CuTe-DSL / 两阶段 split / `ape` / `coff` | `deepseek_v41/**` 零命中或从不 import（`\bape\b` 与 `coff` 均为 **0**） |
| `overlap`（V4.0 的 `self.overlap`） | ⚠️ **不能算零命中**：`deepseek_v41/**` 里 `overlap` 有 **32** 处（如 `attention.py:847`、`nvidia/model.py:158`、`amd/rocm.py:778`）。但那些都是**通用并行概念**（mHC overlap、多流 overlap），**不是** V4.0 compressor 的那个 `self.overlap` —— 该属性在 v4.1 里不存在 |

---

## 3. 注册的算子清单

**重要负面事实（限定范围后成立）**：`deepseek_v41/**` 下 `direct_register_custom_op` / `torch.library` / `register_fake` **全部零命中**；compressor 子系统的算子都是普通 Python 函数，内部直接 launch `@triton.jit` —— 具体是 **compressor 子系统自己的 7 个 `@triton.jit` kernel**（`compressor.py` 1 个 `_ring_slot_mapping_kernel` + `fused_compress_quant_cache.py` 6 个）。这 7 个**没有 CompileKey、没有 warmup**（`attention.py:617-622` 只 warmup 了 `_COMPRESSED_SLOT_MAPPING_KERNEL`，那是相邻子系统的工具 kernel）。

> ⚠️ **不要把这句话推广到整个 `deepseek_v41/**`**：该目录**有**自定义 op 注册与 JIT warmup 基础设施 —— `common/ops/cache_utils.py:1034` 定义 `CompileKey`、`:1163` 有 `get_warmup_keys`、`:1174` 有 `compile()`；`nvidia/ops/fused_wo_a.py:85-97` 定义 `FusedWoAKernel.CompileKey`；`attention.py:653-657` 真的调用 `register_warmup()`。**「零 CompileKey / 零 warmup」只是 compressor 子系统的特征。**

| 名字 | 类型 | 位置 | 作用 | H200 |
|---|---|---|---|---|
| `fused_save_compress_norm` | 入口 | `fused_compress_quant_cache.py:22-114` | 池化 + 门控 + RMSNorm + ring 保存，一个 kernel 全做 | ✅ |
| `_fused_save_compress_norm_kernel` | `@triton.jit` | `:117-212` | 上述主体；两类 program | ✅ |
| `_store_latent` | `@triton.jit` 内联 | `:215-221` | RMSNorm(512) → BF16 latent | ✅ |
| `rope_quant_insert` | 入口 | `:224-300` | 按 `kv_cache.shape[-1]` 选记录 kernel | ✅ |
| `_rope_quant_insert_kernel` | `@triton.jit` | `:303-351` | **584 B 记录**：NoPE fp8 + RoPE bf16 + 7 个 UE8M0 + pad | ✅ **唯一** |
| `_rope_quant_insert_mxfp8_kernel` | `@triton.jit` | `:354-405` | 528 B（先 RoPE 再全 512 维量化） | ❌ SM100 |
| `_rope_quant_insert_nvfp4_kernel` | `@triton.jit` | `:408-461` | 288 B（e2m1 + e4m3 scale，每 16 维） | ❌ SM100 |
| `_rope_plain_insert_kernel` | `@triton.jit` | `:464-506` | 普通 512 元素行（`[448 NoPE / 64 RoPE]`）；`STORE_FP8` 时另取 per-tensor `fp8_scale` | ❌ 需 `use_fp8_ds_mla_layout=False` 的后端 |
| `_ring_slot_mapping_kernel` | `@triton.jit` | `compressor.py:60-84` | ring 槽 `block*CAPACITY + pos%CAPACITY`，block 0 是 null | ✅ |
| `CompressorBackend` | `AttentionBackend` | `compressor.py:31-49` | 声明 head size `[512,1024]`、block size `MultipleOf(1)` | ✅ |
| `CompressorMetadataBuilder` | `AttentionMetadataBuilder` | `compressor.py:87-133` | 造 `CompressorMetadata`；`_cudagraph_support=ALWAYS` | ✅ |
| `CompressorStateCache` | `nn.Module` | `compressor.py:136-172` | FP32 ring 的 spec/绑定；`forward()` 是空 stub | ✅ ratio 2 |
| `indexer_k_norm_rope_store` | 入口 | `common/ops/indexer_k_store.py:29-119` | latent 的下游消费者（indexer K） | ✅ |
| `get_compressed_slot_mapping` | 工具 kernel 类 | `vllm/v1/attention/backends/mla/compressor_utils.py:30-231` | **给主注意力**算压缩 slot；**不是 compressor 本体** | ✅ |

---

## 4. Compressor 类型分类（7 类）

下面 C1–C7 是**按「一段可独立理解的计算」切分的**，不是按文件或类切分。每一类给出：位置、输入、输出、要点。

### C1 · `fused_wkv_wgate` 投影

- **位置**：`compressor.py:205-231`（`MergedColumnParallelLinear` 构造）、`attention.py:932-983`（并行调度）
- **输入**：`hidden_states [T, 5120]` bf16
- **输出**：`kv_score [T, 512*ratio]` **FP32**
- **要点**：
  - 权重是 `wkv` 与 `wgate` 两片拼起来的；`ratio=1` 时只有 `wkv`（gate 尺寸为 0）。
  - GEMM 是**裸 `torch.mm(hidden, W.T, out_dtype=float32)`**（`:954-959`），绕开 Linear wrapper，因此拿不到 bias 与量化支持。**它只在与 `fused_wqa_wkv` 并行时才在辅助流 0 上**（这一行即辅助流 GEMM：`execute_in_parallel(..., enable=T <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD)`，`:973-981`，默认阈值 **1024**；当 `aux_streams is None or not enable` 时退回当前流串行执行，`vllm/utils/multi_stream_utils.py:112-115`）。
  - 权重映射 `nvidia/model.py:1248-1249`（shard0=`wkv`，shard1=`wgate`）。

### C2 · ratio-2 软门控池化 + FP32 ring

- **位置**：`fused_compress_quant_cache.py:117-212`（kernel 主体）
- **输入**：`kv_score [T, 1024]` FP32、`positions [T]`、`block_table`、ring cache
- **输出**：写 ring 的原始行；并在组边界产出 `latent`
- **要点**：
  - 池化就是 §1 那条逐维 softmax 公式，`axis 0` 是组内 token。
  - **ring 是 ratio 2 独有的**：容量 `max(8, 1 << (num_speculative_tokens + 1).bit_length())`，槽位 `block*CAPACITY + pos % CAPACITY`。
  - 一个请求的 ring 生命周期内**只占一个 block**（`CircularBufferSpec`）。

### C3 · ratio-1 恒等路径

- **位置**：同 kernel，另一条分支
- **要点**：**没有 ring、没有门控**，一个 program 对应一个 token，`pooled = raw[t]`，就是一次 512 宽的普通 load。`grid = num_tokens`（ratio 1）vs `num_reqs + cdiv(num_tokens, ratio)`（ratio 2）。

### C4 · RMSNorm → BF16 latent（`_store_latent`）

- **位置**：`fused_compress_quant_cache.py:215-221`
- **输入**：池化后的 FP32 `[512]`
- **输出**：`latent [512]` **BF16**
- **要点**：
  - 方差除数是**硬编码的 512**（`:219`），不是按实际长度算的。
  - C2 与 C3 **共用**它，所以 ratio 1 与 ratio 2 的 latent 统计口径一致。
- **与 V4.0 的差异**：V4.0 的对应逻辑在 CuTe-DSL 里且会加上 `ape`；V4.1 无 APE。**norm → 量化的先后两代一致**（V4.0 同样是先 RMSNorm、再 RoPE、再量化：`v4/nvidia/ops/sparse_attn_compress_cutedsl.py:274-282` → `:301-308`）；真正的差异在记录格式（见 C6）。

### C5 · 环形 slot 映射与 ring 生命周期

- **位置**：`compressor.py:60-84`（kernel）、`:104-133`（builder）、`:136-172`（spec）
- **输入**：`block_table [num_reqs, max_blocks]` int32（**ring 组自己的**）、`positions [T]`、`token_to_req_indices [T]`、`num_actual_tokens`
- **输出**：`CompressorMetadata{ slot_mapping, ... }`
- **要点**：
  - slot = `block * CAPACITY + pos % CAPACITY`，block 0 表示 null，所以真实槽位从 1 开始编。
  - **V4.0 的做法不同**：V4.0 的 `state_cache` 用的是 `SlidingWindowMLASpec`，其 slot **直接取自通用的 `common_attn_metadata.slot_mapping`**（`v4/compressor.py:127`），**不是** `get_compressed_slot_mapping` —— 后者只服务 sparse-MLA builder 与 indexer。
  - `CompressorStateCache.forward` 是空 stub（`:169`）；`self.max_num_reqs` 赋值后从未使用（`:217`）；**`rotate` 同样是死参数** —— 构造时传 `True`（`attention.py:582`）、存进 `self.rotate`（`compressor.py:205`），但 `deepseek_v41/**` 内**从不读取**。

### C6 · `rope_quant_insert`：584 B 记录写入

- **位置**：`fused_compress_quant_cache.py:224-300`（入口）、`:303-351`（584 B kernel）
- **输入**：`latent [T, 512]` BF16、`positions`、主压缩 cache（`uint8`）
- **输出**：主压缩 cache 里一条条 584 B 记录
- **要点**：
  - 分派**看 `kv_cache.shape[-1]`**：584 选 `_rope_quant_insert_kernel`；528 / 288 在 H200 上取不到（见 §2「到不了的分支」）。
  - RoPE 用的是**组起点**的位置，不是 token 自己的位置。
  - 记录内部：`[0,448)` fp8 e4m3 的 NoPE 维、`[448,576)` 64 维 RoPE 用 **bf16**、`[576,583)` 7 个 UE8M0 缩放、`[583]` 强制 0。
  - **记录在页里不是连续的**：`values = page + (slot % CACHE_BLOCK)*576`，`scales = page + CACHE_BLOCK*576 + (slot % CACHE_BLOCK)*8`。

### C7 · 一个 latent 供两个下游用（与 indexer 的配合）

- **位置**：`attention.py:711-715`（`forward_kv`）与 `:868-903`（两级并行）；indexer 侧 `:1405-1437`
- **输入**：`latent [T,512]` BF16（只读）；`positions [T]`；`self.rotary_emb`
- **输出**：无返回；写**两个不同的** cache：主压缩 cache（584 B/记录）与 indexer K cache（132 B/行）
- **要点**：
  - 两条路只读 latent，所以可以并行（`maybe_execute_in_parallel`）。
  - **H200 上两条路共用同一个 `cos_sin_cache`**：`attention.py:416` 是 `self.indexer_rotary_emb = self.rotary_emb`，是别名而不是第二个实例。
  - indexer 侧还会 `wk -> k_norm(128) -> RoPE`，与主路完全独立。

---

## 5. 模块位置与作用

```
唯一拥有 compressor 的层：layer 2 / 8 / 14 / 20（kv_source_layer_ids）；其余 38 层只是复用别人的压缩 cache

layer 0,1      ratio 0  无 compressor、无 indexer、无压缩 KV
layer 2        ratio 2  compressor（wkv+wgate+norm）+ indexer(owns_k)          ← kv SOURCE
layer 3..7     ratio 2  无自己的 compressor，复用 layer 2 的压缩 cache
layer 8        ratio 2  compressor（同上）                                     ← kv SOURCE
layer 9..13    ratio 2  复用 layer 8
layer 14       ratio 2  compressor（同上）                                     ← kv SOURCE
layer 15..19   ratio 2  复用 layer 14
layer 20       ratio 1  compressor（wkv+norm，**无 gate**）+ indexer(owns_k)   ← kv SOURCE + candidate
layer 21..23   ratio 1  复用 layer 20；无 indexer
layer 24       ratio 1  只有 indexer（共享 layer 20 的 K cache）；无 compressor
layer 25..27   ratio 1  复用 layer 20；无 indexer
layer 28       ratio 1  只有 indexer
layer 29..31   ratio 1  复用 layer 20；无 indexer
layer 32       ratio 1  只有 indexer
layer 33..35   ratio 1  复用 layer 20；无 indexer
layer 36       ratio 1  只有 indexer
layer 37..39   ratio 1  复用 layer 20；无 indexer
layer 40..42   ratio 0  MTP：无 compressor、无 indexer
```

**两个 SWA cache 写入者要分清**：常规 `forward` 走 C++ 融合算子 `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`（`attention.py:1072`），一次完成 q/kv 的 norm+RoPE+量化+插入；而 `forward_kv` 路径用更窄的 `_C.fused_deepseek_v4_kv_rope_insert`（`:699`），只做 kv 的 RoPE+插入。两者写的是**同一个** SWA cache。

**`forward_kv` 的唯一调用者**是 replay 路径：`nvidia/model.py:659` 的 `DeepseekV4DecoderLayer.write_kv`（`:627`）。**dspark 不走 `forward_kv`**，它用自己的 `dspark.py:199` → `_insert_context_kv`（`:254`）。

**单个 kv-source 层的包含树**（`compressor.py:185-248`）：

```
layers.<L>.attn                        DeepseekV4Attention（L ∈ 2/8/14/20）
├── fused_wqa_wkv                      MergedColumnParallelLinear，量化
├── compressor                         DeepseekCompressor           <=== 本子系统
│   ├── fused_wkv_wgate                MergedColumnParallelLinear，quant_config=None（bf16）
│   ├── norm                           RMSNorm(512, eps=1e-20)
│   ├── state_cache                    CompressorStateCache（仅 ratio > 1）
│   └── insert_cache()                 RoPE + 量化 + 写主 cache
├── swa_cache                          DeepseekV4SWACache（另一个 cache，每层都有）
└── indexer                            DeepseekV4Indexer（仅 index_source 层）
    ├── wk                             ReplicatedLinear(512, 128)，quant_config=None
    ├── k_norm                         RMSNorm(128)
    ├── k_cache                        DeepseekV4IndexerCache，H200 上 132 B/行
    └── _produce_k()                   消费 compressor 的 latent
```

**哪些地方没有 compressor**：

| 地方 | 结论 |
|---|---|
| `swa_cache` | 属 SWA 分支，不是 compressor |
| `indexer.proj` / `wk` / `k_norm` | 属 indexer；`wk` 只是 latent 的**消费者** |
| `is_corpus` / `_produce_latent` / `forward_cached` | **全树零命中** |
| `self.indexer.compressor` 属性 | **零命中**（唯一 `indexer.compressor` 字面是 `amd/rocm.py:614` 的注释，说 V4.1 **没有**它） |
| `ape` / `coff` | `deepseek_v41/**` 连字面都零命中（`\bape\b` 与 `coff` 均为 **0**） |
| `overlap` | ⚠️ **不是零命中**：`deepseek_v41/**` 有 **32** 处（如 `attention.py:847`、`nvidia/model.py:158`）。但它是通用并行概念（mHC overlap、多流 overlap），**不是** V4.0 compressor 的那个 `self.overlap` |
| `use_fp4_cache` 这个名字 | ⚠️ **是两个不同的东西，不要混**：① **V4.0 compressor 的构造参数** —— `v4/compressor.py:220` 就是 `DeepseekCompressor.__init__` 的形参（`:229` 存储、`:296-298` 断言、`:492` 传 kernel；调用点 `v4/attention.py:1030`）；② V4.1 里同名的**索引器**参数 —— `indexer_k_store.py:38` / `attention.py:1393`。**V4.1 的 `DeepseekCompressor.__init__` 没有这个参数**（`compressor.py:185-194` 的签名里没有） |
| `direct_register_custom_op` / `torch.library` | `deepseek_v41/**` 零命中 |
| `deepseek_v41/cpu/`、`deepseek_v41/xpu/` | **不存在**（只有 `amd/`、`nvidia/`、`common/`） |

---

## 6. 跨层拓扑：发布/消费 + latent 分叉

**发布/消费**：只有 kv-source 层真正拥有主压缩 cache；其余 ratio>0 的层通过 `_static_forward_context` 拿到同一个张量。

**latent 分叉**：`compressor.forward` 出 latent 后，两条下游（主 cache 写入、indexer K 写入）**都只读它**，因此可以并行；它们在 `_sparse_indexer_and_attn` 里被 join。

---

## 7. Shape / Dtype / 记录格式总表

### 7.1 数值与索引通路

| 张量 | shape | dtype | 约束 / 说明 |
|---|---|---|---|
| `hidden_states` | `[T, 5120]` | bf16 | 模型输入 |
| `fused_wkv_wgate.weight` | `[512*ratio, 5120]` | **bf16** | 检查点里是 `wkv`/`wgate` 两片 |
| `kv_score` | `[T, 512*ratio]` | **FP32** | GEMM 直接出 FP32；前 512 是 kv，后 512 是门控分数 |
| `latent` | `[T, 512]` | **BF16** | pre-RoPE；只在组边界有效 |
| 主压缩 cache | `[num_blocks, 64, 584]` `uint8` | packed fp8+bf16 | 见 §7.2 |
| `state_cache` ring | `[num_blocks, CAPACITY, 1024]` `float32` | **FP32** | 4096 B/行；仅 ratio 2 |
| indexer `k_cache` | `[num_blocks, block_size // compress_ratio, 132]` `uint8` | fp8 + scale | dim1 是 `num_states`（cr2=32、cr1=64），**不是** `block_size` |
| SWA cache | `[num_blocks, 32, 584]` `uint8` | 同主压缩 cache | 每层都有；block_size=32 |

### 7.2 记录/格式选择矩阵（★）

| 记录宽度 | 适用 | 布局 | kernel | 前提 | H200 |
|---|---|---|---|---|---|
| **584 B** | V4.1 fp8_ds_mla | 448 B NoPE fp8 + 128 B bf16 RoPE + 7×(1 B UE8M0) + 1 B pad | `_rope_quant_insert_kernel`（`:303-351`） | `use_fp8_ds_mla_layout=True` | ✅ **唯一** |
| 528 B | V4.1 MXFP8 | 512 fp8（含 RoPE 维）+ 16 UE8M0（每 32 维） | `_rope_quant_insert_mxfp8_kernel`（`:354-405`） | 需 `kv_mxfp8`（SM100） | ❌ |
| 288 B | V4.1 NVFP4 | 256 B e2m1 对 + 32 B e4m3 scale（每 16 维） | `_rope_quant_insert_nvfp4_kernel`（`:408-461`） | 需 `nvfp4_ds_mla` | ❌ 启动 `ValueError` |
| **512 元素**（**不是 516 B**） | plain 行（bf16 / per-tensor fp8） | 512 个**元素**的行：bf16 = 1024 B、fp8 = 512 B；`fp8_scale` 是**独立的 1 元素 fp32 缓冲**，不是记录字段 | `_rope_plain_insert_kernel`（`:464-506`） | 需 `use_fp8_ds_mla_layout=False` 的后端 | ❌ |

分派代码（`:254-259`）：`{584:…, 528:…, 288:…}.get(kv_cache.shape[-1])`，取不到就 assert 失败。

**584 B 的精确字节偏移**（由 kernel 反推，不是抄自散文）：

| 记录内偏移 | 字节 | 内容 | 代码 |
|---|---|---|---|
| `0 .. 447` | 448 | fp8 **e4m3** 的 NoPE 维 0..447，分 7 个量化块 × 64 维 | `:303-351` |
| `448 .. 575` | 128 | 64 个 RoPE 维（448..511），**bf16**，不加缩放 | 同上 |
| `576 .. 582` | 7 | 7 个 **UE8M0** 缩放，每块一个 | 同上 |
| `583` | 1 | 强制 0（`SANITIZE_CACHE_NANS=False` 时的填充） | 同上 |

### 7.3 页几何（`alignment=576`、`state_content_bytes=584`）

算法（`kv_cache_interface.py:227-234, 544-560, 676-682`）：`num_states = block_size // tokens_per_state`；`unpadded = num_heads × num_states × state_content_bytes`；`page = round_up(unpadded, alignment)`。

| `block_size` | `ratio` | 每页状态数 | 未对齐 | 对齐后页 | 倍数 |
|---|---|---|---|---|---|
| 64 | 2 | 32 | 18688 | **19008** | 33 |
| 64 | 1 | 64 | 37376 | **37440** | 65 |
| 128 | 2 | 64 | 37376 | 37440 | 65 |
| 128 | 1 | 128 | 74752 | 74880 | 130 |

⚠️ **`37376 = 64×576 + 512`，所以 ratio 1 的页也要补齐到 37440。** sm90 上 `block_size` 自动解析为 **64**（`sparse_mla.py:88-90`），**128 的两行只在 SM100 可达**。

`state_cache` ring 的页（`CircularBufferSpec`）是 **4096 B/行**（1024 个 fp32），与上面四种都不同。

### 7.4 ★「两个不同的 cache」必须分清（实为三层 + 一个 ring）

| cache | 谁写 | H200 宽 | dtype | 谁拥有 | 前缀 |
|---|---|---|---|---|---|
| **主压缩 KV cache** | `insert_cache()` → `_rope_quant_insert_kernel` | **584 B/记录** | uint8 | **仅 kv-source** | `layers.<L>.attn` |
| **SWA cache** | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`（C++ op，`attention.py:1072`；`forward_kv` 走更窄的 `fused_deepseek_v4_kv_rope_insert`，`:699`） | **584 B/记录** | uint8 | **每一层** | `…attn.swa_cache` |
| **indexer K cache** | `indexer_k_norm_rope_store()` | **132 B/行** | uint8 | 仅 kv-source（`owns_k`） | `…attn.indexer.k_cache` |
| `state_cache` ring | `_fused_save_compress_norm_kernel` | 4096 B/行 | **FP32** | 仅 ratio>1 的 kv-source | `…attn.compressor.state_cache` |

**关键区别**：主压缩 cache 与 SWA cache 在 H200 上**恰好都是 584 B/记录**（`attention.py:513` vs `:516-518`），但它们是不同张量、不同 module、不同前缀、不同 writer，而且 **SWA 的 block_size = 32、压缩 cache 的 block_size = 64**（`:564` vs `:1150`）。只有 `nvfp4_ds_mla` 才会让两者宽度分叉，而它在 sm90 被拒（`:523-527`）。

两个 cache 在**稀疏注意力取数**阶段被合并：`combine_topk_swa_indices`（`common/ops/cache_utils.py:925-971`，kernel `:1050-1134`）输出每行 `[压缩 topk_len 个][SWA swa_len 个][-1 填充]`，宽度 `round_up(topk+window, 128)`；源码注释说明 `B_TOPK` 对 h_q=64 是 **64**、对 h_q=128 是 **128**，而对齐常量 `_SPARSE_PREFILL_TOPK_ALIGNMENT = 128`（`cache_utils.py:917-922`）。另有 `combined_lens = topk_len + swa_len`。**compressor 本身不参与这个合并。**

### 7.4b ★ H200 页几何与 per-token 成本（容量规划必需）

**四个 spec 在 H200（`fp8_ds_mla`、`cache_config.block_size=64`）上的实算**：

| cache | spec / 关键字段 | 每页状态数 | 未对齐 | **对齐后页** |
|---|---|---|---|---|
| 主压缩 KV（cr2） | `MLAAttentionSpec(tokens_per_state=2, state_content_bytes=584, alignment=576)`（`attention.py:1149-1163`） | 64/2 = **32** | 32×584 = 18 688 | **19 008** |
| 主压缩 KV（cr1，layer 20） | 同上但 `tokens_per_state=1` | **64** | 64×584 = 37 376 | **37 440** |
| **indexer K cache（cr2）** | `MLAAttentionSpec(tokens_per_state=2, head_size=132, alignment=576)`（`attention.py:1219-1239`） | **32** | 32×132 = 4 224 | **4 608** |
| **indexer K cache（cr1）** | 同上 `tokens_per_state=1` | **64** | 64×132 = 8 448 | **8 640** |
| **SWA cache（每层）** | `SlidingWindowMLASpec(alignment=576, state_content_bytes=584)`（`sparse_swa.py:124-140`）；`block_size=32` 硬编码于 `attention.py:564` | **32** | 32×584 = 18 688 | **19 008** |

> 两个要点：
> 1. **SWA 页与 cr2 压缩页恰好都是 19 008 B**（都是 32 行 × 584 B 再补齐到 576 的倍数），但它们是不同张量（`attention.py:564` 传 `block_size=32`，`:1150` 传 `cache_config.block_size=64`）。
> 2. **同一 kv-sharing 组内所有 cache 共享一个物理 block，block stride 是组内各页之和**（源码注释：*"The indexer page shares a physical block with the main KV cache (the block stride is the sum of every page in the group)"*，`attention.py:1225-1228`；落实代码是 `kv_cache_interface.py:1252-1254` 的 `sum(...)`）。所以这 4 个数**不能各自当作独立页**来估显存。

**`block_stride_alignment`**：`math.lcm(512, page_alignment)` = **`lcm(512, 576) = 4608`**，且**仅当 `self.sparse_logits and not self.rocm_mxfp4` 同时成立**（`attention.py:1234-1238`；H200 上 `rocm_mxfp4` 恒 False，所以实际由 `sparse_logits` 决定）。H200 上 `indexer_sparse_logits` **默认关闭**（`vllm/config/attention.py:94`），所以这个字段是 `None`。

**per-token 字节成本（H200）**：

| 项 | cr2 | cr1 |
|---|---|---|
| 主压缩 KV | 584/2 = **292 B/token** | **584 B/token** |
| indexer K | 132/2 = **66 B/token** | **132 B/token** |
| SWA（每层） | **584 B/token** | **584 B/token** |

### 7.4c ★ 主注意力的 compressed slot mapping

`get_compressed_slot_mapping` 不只服务 indexer —— **主注意力（`sparse_mla.py`）也用它**，触发条件是 ratio。真码（逐字，`sparse_mla.py:194-205`；`v4/sparse_mla.py` 同构）：

```python
slot_mapping = cm.slot_mapping
if self.compress_ratio > 1:
    slot_mapping = get_compressed_slot_mapping(
        cm.num_actual_tokens,
        cm.slot_mapping,
        cm.query_start_loc,
        cm.seq_lens,
        cm.block_table_tensor.clamp_(min=0),
        int(self.kv_cache_spec.num_states),
        self.compress_ratio,
        out=self.compressed_slot_mapping_buffer,
    )
```

也就是**先无条件取 `cm.slot_mapping`，再在 ratio > 1 时覆盖** —— 没有 `else`（`cm` 是 `common_attn_metadata` 的局部别名，见 `sparse_mla.py:191`）。

**这条分支正是 `Compressor.forward` 能安全走 `state_cache is None` 的原因**：ratio 1 时压缩 cache 与 token 一一对应，不需要额外映射。

这个 kernel **同时是 DCP/CP 感知的**（`compressor_utils.py:60-106, 177-228`，含 `dcp_world_size` / `cp_interleave`），所以 `deepseek_v41/**` 下**至少还有一条 H200 可达路径落在 §4 的 C1–C7 之外**：**DCP 切分下的压缩 slot 映射**。§4 的「无遗漏」自评应据此理解为「compressor 自身算子无遗漏；压缩 slot 映射的 DCP 变体属相邻子系统，没列在 C1–C7 里」。

### 7.4d ★ ring 对调度器可见的两个事实

`CircularBufferSpec` 除 `block_size` 外还有两个对调度器可见的字段（`kv_cache_interface.py:919-921, 939-941`）：

| 字段 | 值 | 含义 |
|---|---|---|
| `block_table_token_alignment` | `None` | 本 cache 的 block table 不做 token 级对齐，调度器不必按 token 粒度切块 |
| `prefix_cacheable` | `False` | 本 cache 不可前缀缓存 —— ring 内容随 position 滚动，命中前缀没有意义 |

这两个字段合起来说明：**调度器知道 ring 与主 cache 不同、也不参与 prefix caching**，所以不会对它做前缀复用或 token 级 block 对齐。

### 7.5 config.json → 代码的对应（只列 compressor 关心的）

| config（`text_config.*`） | 值 | 读它的代码 |
|---|---|---|
| `head_dim` / `qk_rope_head_dim` | **512** / **64** | `compressor.py:212-214`（有 assert） |
| `hidden_size` / `rms_norm_eps` | **5120** / **1e-20** | `compressor.py:203, 215` |
| `compress_ratios` | 43 项 = `[0,0] + [2]×18 + [1]×20 + [0]×3` | `attention.py:310-321`（非 {0,1,2} 抛 `ValueError`） |
| `kv_source_layer_ids` | **[2,8,14,20]** | `attention.py:322-324, 333, 343-345` |
| `index_source_layer_ids` | **[2,8,14,20,24,28,32,36]** | `attention.py:325-327, 334, 346-348` |
| `rope_theta` / `compress_rope_theta` | **10000** / **160000** | `common/rope.py:29`（分界 `compress_ratio > 0`） |
| `index_topk` | **512** | indexer 的 top-k |
| `candidate_source_layer_id` | **20** | 候选块来自 ratio 1 的那个 kv SOURCE 层 |
| `candidate_topk_blocks` | **2048** | 每步最多保留的候选块数；`candidate_block_buffer` 的宽度就是这个值（`nvidia/model.py:717-721`） |
| `candidate_block_size` | **8** | 一个候选块的 token 数 |
| `indexer_sparse_logits` | **`False`**（默认） | 只有打开它，`block_stride_alignment` 才非 `None`（见 §7.4b） |

---

## 8. 与上一代 DeepSeek-V4 的差异 + 容易混淆的同名项

### 8.1 逐项对照

| 项 | V4.0（`deepseek_v4/`） | V4.1（`deepseek_v41/`） | 备注 |
|---|---|---|---|
| 压缩池化 | 同样是逐维 softmax 门控 | 同 | 两代一致的语义 |
| 记录格式 584 B | 有（`cutedsl` 路径） | 有（Triton 路径） | 两代共享，可互读 |
| APE | 有（`ape`） | **无**（`\bape\b` 零命中） | 这是真实差异 |
| 实现载体 | CuTe-DSL / `save_partial_states` | Triton kernel + Python 调度 | v41 里没有 cutedsl 文件被 import |
| cache 写入 | `forward` 内部自己写 | 拆出独立的 `insert_cache()` | 接口差异 |
| `state_cache` 的 slot | `SlidingWindowMLASpec` + 通用 `slot_mapping`（`v4/compressor.py:127`） | `CircularBufferSpec` + `get_compressed_slot_mapping` | 机制不同 |
| `use_fp4_cache` | **有**这个构造参数（`v4/compressor.py:220`） | `DeepseekCompressor.__init__` **没有**它；记录选择改由 `attention.py:132-173` 的 `_resolve_dsv4_kv_cache_dtype` + `:512-522` 的字节宽推导 + `fused_compress_quant_cache.py:254-284` 的**运行时按 cache 张量分派**共同决定。`v41/compressor.py` 全文**没有 `kv_cache_dtype` 或 `use_fp4_cache`** | ⚠️ 名字相同但不是同一个东西，另有索引器也用了 `use_fp4_cache` |

### 8.2 同仓库其他同名概念（注意区分）

| 同名/近名 | 在哪 | 是什么 | 与 compressor 的关系 |
|---|---|---|---|
| `DeepseekV4Indexer.compressor` | 只在 `amd/rocm.py:614` 的注释里 | 注释说 V4.1 **没有**这个属性 | 无关 |
| `indexer_k_norm_rope_store` | `common/ops/indexer_k_store.py` | 把 latent 变成 indexer K | **下游消费者** |
| `use_fp4_cache` | `indexer_k_store.py:38` | 索引器自己的 fp4 开关 | 与 V4.0 compressor 的同名参数**不是一个物** |
| `get_compressed_slot_mapping` | `mla/compressor_utils.py` | 压缩坐标系下的 slot 换算 | 服务主注意力与 indexer，**不是** compressor 本体 |
| `CompressorBackend` / `CompressorMetadataBuilder` / `CompressorStateCache` | `compressor.py` | ring 的 backend / builder / spec | 属 compressor |

---

## 9. 结构图

下面 4 张图都由 `compressor\build_doc.py` 从 `compressor\make_compressor_diagram.py` 的产物里直接取用。制图器在**终端显示宽度**（汉字按 2 格算）上排版，然后自动审计；五条判据全部为 0 问题：

1. 每个 `▼` 的正上方、每个 `▲` 的正下方必须是连接符；
2. 不得有悬空的 `|`（四邻至少一个是盒线字符）；
3. 两条线相交时按盒线规则归并（`-` × `|` → `+`），而不是互相覆盖；
4. **每个 `▼` 必须落在它正上方那个盒的中线上**（左右偏移 ≤ 1 格）；
5. 用 ASCII 盒线而不是 Unicode 框线，所以在任何终端、任何字体下都能对齐。

另外 6 张图（模块包含树、584 B 字节排布、cache 家族、dtype 阶梯等）见 `compressor\COMPRESSOR_diagram.md`。

### 9.1 compressor 存在的层

```text
  唯一的 compressor 拥有者是 layer 2 / 8 / 14 / 20（kv_source_layer_ids）；其余 38 层只用别人的压缩 cache
  
  layer 0,1      ratio 0  无 compressor、无 indexer、无压缩 KV
  layer 2        ratio 2  compressor（wkv+wgate+norm）+ indexer(owns_k)   ★ kv SOURCE
  layer 3..7     ratio 2  无自己的 compressor，复用 layer 2 的压缩 cache
  layer 8        ratio 2  compressor + indexer(owns_k)                    ★ kv SOURCE
  layer 9..13    ratio 2  复用 layer 8
  layer 14       ratio 2  compressor + indexer(owns_k)                    ★ kv SOURCE
  layer 15..19   ratio 2  复用 layer 14
  layer 20       ratio 1  compressor（wkv+norm，无 gate）+ indexer(owns_k) ★ kv SOURCE + candidate
  layer 21..23   ratio 1  复用 layer 20；无 indexer、无 compressor
  layer 24       ratio 1  只有 indexer（共享 layer 20 的 K cache）；无 compressor
  layer 25..27   ratio 1  复用 layer 20；无 indexer、无 compressor
  layer 28       ratio 1  只有 indexer（同上）
  layer 29..31   ratio 1  复用 layer 20；无 indexer、无 compressor
  layer 32       ratio 1  只有 indexer（同上）
  layer 33..35   ratio 1  复用 layer 20；无 indexer、无 compressor
  layer 36       ratio 1  只有 indexer（同上）
  layer 37..39   ratio 1  复用 layer 20；无 indexer、无 compressor
  layer 40..42   ratio 0  MTP：无 compressor、无 indexer
```

> 与 §5 的逐层表完全一致：唯一的 compressor 拥有者是 layer 2 / 8 / 14 / 20，`★ kv SOURCE` 标出它们；其余 38 层只是复用别人的压缩 cache。

### 9.2 一层 forward 的执行流（C1 在这里）

```text
  +-----------------------------------------------------------------------------------------------+
  | hidden_states  [T, 5120] bf16   （T = 本步 token 数）                                         |
  +-----------------------------------------------------------------------------------------------+
                                                  |
                                                  |
                                                  ▼
  +-----------------------------------------------------------------------------------------------+
  | 三路并行投影（attention.py:932-983）                                                          |
  |   主 流        fused_wqa_wkv(hidden)           -> qr_kv [T, 1792]                             |
  |   辅助流 0     torch.mm(hidden, W_kv_gate.T)   -> kv_score [T, 512*ratio] FP32                |
  |   辅助流 1     indexer.weights_proj(hidden)    -> indexer_weights                             |
  |   仅当 T <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD（默认 1024）才真正并行                     |
  +-----------------------------------------------------------------------------------------------+
                                                  |
                                                  |
                                                  ▼
  +-----------------------------------------------------------------------------------------------+
  | 汇合点 1  execute_in_parallel：qr_kv / kv_score / indexer_weights 全部就绪                    |
  +-----------------------------------------------------------------------------------------------+
                                                  |
                                                  |
                                                  ▼
  +-----------------------------------------------------------------------------------------------+
  | q 通路  qr_kv -> _split_qkv_and_norm -> qr, kv                                                |
  |           -> _fused_qnorm_rope_kv_insert：写 SWA cache（第 2 个 cache，每层都有，584 B/记录） |
  |           -> q = wq_b(qr) 供稀疏注意力使用                                                    |
  +-----------------------------------------------------------------------------------------------+
                                                  |
                                                  |
                                                  ▼
  +-----------------------------------------------------------------------------------------------+
  | kv 通路  kv_score -> compressor(kv_score, positions) -> latent [T, 512] BF16                  |
  |           ★ FP32 -> BF16 的落差就发生在这里（即下文的 C4 步）                                 |
  +-----------------------------------------------------------------------------------------------+
                                                  |
                                                  |
                                                  ▼
  +-----------------------------------------------------------------------------------------------+
  | 汇合点 2  maybe_execute_in_parallel：两条下游都只读 latent                                    |
  |   insert_cache(latent)        -> 主压缩 KV cache（第 1 个 cache，584 B/记录）                 |
  |   indexer._produce_k(latent)  -> indexer K cache（第 3 个 cache，132 B/行）                   |
  |   随后 _sparse_indexer_and_attn：所有 cache 写入都已汇合，才允许读                            |
  +-----------------------------------------------------------------------------------------------+
```

> 对应 §4 的 C1：`hidden_states` 先进三路并行投影，`kv_score` 由辅助流 0 的裸 `torch.mm` 产出；两条 lane（q 通路 / kv 通路）在「汇合点 2」汇合，之后两个消费者都只读 latent。框内标注的门控条件见 §2 的 G 矩阵。

### 9.3 ratio-2 融合 kernel 的状态机（C2/C4 在这里）

```text
  +--------------------------------------------+    +-----------------------------------------------+
  | 请求 program   pid ∈ [0, num_reqs)         |    | 分组 program   pid >= num_reqs                |
  |   读 ring[(position-1) % capacity]         |    |   t = (pid-num_reqs)*ratio + (pos[t]+1)%ratio |
  |   读 raw[start]（本 chunk 首 token）       |    |   若 t == query_start_loc[req] 则直接返回     |
  |   池化 -> RMSNorm -> latent[start]         |    |   读 raw[t-1], raw[t]                         |
  |   然后写回本 chunk 末 min(len,capacity) 行 |    |   池化 -> RMSNorm -> latent[t]                |
  +--------------------------------------------+    +-----------------------------------------------+
                         |                                                  |
                         |                                                  |
                         ▼                                                  ▼
  +--------------------------------------------+    +-----------------------------------------------+
  | 读 ring                                    |    | 读 raw                                        |
  |   slot = slot_mapping[start]               |    |   两行都在本 chunk 内                         |
  |   row = (position-1) % capacity            |    |                                               |
  +--------------------------------------------+    +-----------------------------------------------+
                         |                                                  |
                         |                                                  |
                         ▼                                                  ▼
  +--------------------------------------------+    +-----------------------------------------------+
  | tl.debug_barrier()                         |    | _store_latent（C4）                           |
  |   所有 lane 都已读完 ring                  |    |   variance = sum(x*x)/512 -> bf16             |
  +--------------------------------------------+    +-----------------------------------------------+
                         |                                                  |
                         |                                                  |
                         ▼                                                  ▼
  +--------------------------------------------+    +-----------------------------------------------+
  | 写回 ring 尾部（此刻才写）                 |    | -> latent[组边界] BF16 [512]                  |
  |   绝不能与上面的读竞争                     |    |   只有组边界 token 发布                       |
  +--------------------------------------------+    +-----------------------------------------------+
                         |
                         |
                         ▼
  +-------------------------------------------------------------------------------------------------+
  | ratio 1：无 ring、无门控 —— 一个 program 一个 token，pooled = raw[t]（一次 512 宽 load）        |
  | grid：ratio 1 = num_tokens；ratio 2 = num_reqs + cdiv(num_tokens, ratio)                        |
  +-------------------------------------------------------------------------------------------------+
```

> 对应 §4 的 C2 与 C4：左列是「请求 program」（负责收尾本 chunk 首 token 结束的那个组），右列是「分组 program」。两列的箭头各自落在本列盒子的中线上，所以可以逐行对照。最后一行的两列合一说明 ratio 1 的退化路径。

### 9.4 一个 latent 分给两个下游（C7 在这里）

```text
  +---------------------------------------------------------------------+
  | compressor.forward(kv_score, positions) 返回 latent [T, 512] BF16   |
  | latent 是 pre-RoPE 的，且只读：下游没有任何一处会改它               |
  +---------------------------------------------------------------------+
                                     |
                                     |
                                     ▼
  +------------------------------------------------------------------------------------------------------------------------------------------+
  | latent 的分叉点：两条下游都只读它，                                                                                                      |
  | 所以 attention 层可以把它们并行调度                                                                                                      |
  | （maybe_execute_in_parallel）                                                                                                            |
  +------------------------------------------------------------------------------------------------------------------------------------------+
                                                                         |
                                     +-----------------------------------+------------------------------------+
                                     ▼                                                                        ▼
  +---------------------------------------------------------------------+      +-------------------------------------------------------------+
  | 主压缩 KV cache（第 1 个 cache）                                    |      | indexer K cache（第 3 个 cache）                            |
  |   insert_cache(latent, positions, self.rotary_emb)                  |      |   indexer._produce_k(latent, positions, indexer_rotary_emb) |
  |   GPT-J RoPE 用「组起点」位置，再做 UE8M0 fp8 量化                  |      |   wk -> k_norm(128) -> RoPE -> fp8 / mxfp4                  |
  |   -> H200 上 584 B / 记录                                           |      |   -> H200 上 132 B / 行                                     |
  +---------------------------------------------------------------------+      +-------------------------------------------------------------+
                                     |
                                     |
                                     ▼
  +---------------------------------------------------------------------+
  | H200 上两条路共用同一个 cos_sin_cache —— attention.py:416 是        |
  | `self.indexer_rotary_emb = self.rotary_emb`，是别名而不是第二个实例 |
  +---------------------------------------------------------------------+
```

> 对应 §4 的 C7 与 §6：latent 是分叉点，两条下游都只读它，因此可以并行；末框点出 H200 上两条路共用同一个 `cos_sin_cache`（`attention.py:416` 是别名）。


---

## 10. 单元测试覆盖情况

主文件：`tests/kernels/test_compressor_kv_cache.py` = **2423 行、25 个 `def test_*`**，没有模块级 `pytestmark`，每个测试自带 skip。H200 上 **20 个运行、5 个跳过** —— 跳过的是 `:978`、`:1111`、`:2042`、`:2157`、`:2290` 这五个**函数**（`is_rocm()` / `_on_gfx950()` 门控，后者在 sm90 上必然为真）。`:1752` 的 `use_fp4=True` 是**参数级**而不是函数级，它只跳过那一个参数组合，整个函数仍然运行。另外有 **8 个测试完全没有 skip 装饰器**，其中 `:1889` 用函数体内的 `pytest.importorskip("cutlass")` 做运行时跳过。

> ⚠️ **头部 import 来自 `vllm.models.deepseek_v4`（V4.0）**（`:21-31`），v4.1 的 kernel 是在各测试体内**局部 import** 的。「文件里出现 v4.1 符号」≠「覆盖了 v4.1 路径」。

### 10.1 真正覆盖 compressor 的用例

| 用例 | 行 | 覆盖 | 断言 |
|---|---|---|---|
| `test_v41_fused_save_compress_and_insert` | **170** | **C2+C4+C6**：`fused_save_compress_norm` + `rope_quant_insert`，cr∈{1,2} × `use_graph` × 5 组 `lengths`；cache 构造成 584（`:211-213`） | 逐字节比对 584 B 记录 |
| C6 记录格式（528 / 288 / plain） | `:1588, 1889` | 三种非 584 记录 | exact / 半格 / `rtol=0.13` |
| `indexer_k_store` 的覆盖 | `tests/kernels/test_fused_indexer_q_rope_quant.py` 的 `:359`、`:444`、`:497`（到 `:573`） | C7 的下游 | — |

### 10.2 覆盖缺口（逐条核实，含被推翻的旧假设）

| # | 缺口 | 结论 |
|---|---|---|
| G1 | 「584 B 没有 v4.1 专属测试」 | **被推翻**（`:170` 就是，逐字节比对）。**残留**：`SANITIZE_CACHE_NANS=True` 在 CUDA 上是死代码（`:11-14, 273, 337`），唯一测试 `:1111` 被 gfx950 skip |
| G2 | `forward` 未用真实权重跑过 | **部分成立且更强**：方法体确实被真实调用（`:840`），但 **`DeepseekCompressor.__init__`（`compressor.py:185-248`）从未被任何测试调用** → ratio 校验、`head_dim/rope_head_dim` assert、`MergedColumnParallelLinear` 构造、`CompressorStateCache` 构造、ratio-1 的 `state_cache is None` 分支、非 dict metadata 返回 `None` **全部零覆盖**；测试里 `norm.weight` 是 `torch.ones` |
| G3 | `insert_cache` 未被直接测试 | **成立**（测试文件里 `compressor.insert_cache` / 它的 monkeypatch 共出现 **5** 次：`:2199, 2200, 2204, 2276, 2284`，全是 stub 或 patch），但真实方法体经 `_prepare_and_attn:896` 间接执行；未覆盖 `latent is None`、`_flashinfer_fp8_kv_scale`、dict 断言 |
| G4 | `CompressorBackend` / `get_kv_cache_spec` 无断言 | **成立**：`CompressorBackend` 在 `tests/` 零出现 |
| G5 | 测试文件 2423 行，25 个测试 | **成立**（已重算） |
| G6 | 覆盖缺口的「最后一句」没有锚点 | 见 §10.3 |
| G7 | `use_graph=True` 之外没有 CUDA graph 覆盖 | **成立** |
| G8 | MTP 层（40–42）零覆盖 | **成立**（ratio 0，本就不该有 compressor） |
| G9 | 没有测试 `_ON_GFX950=True` 的 ROCm tile 分支 | **成立**：`swa_visible.py` 的 15 处 `skipif` 全是 `not cuda.is_available()`，没有 ROCm 门控 |

### 10.3 「默认路径反而覆盖更薄」的四个案例

1. `DeepseekCompressor.__init__` 零调用（见 G2）。
2. `insert_cache` 只有 stub/patch（见 G3）。
3. `CompressorBackend.get_kv_cache_spec` 零断言（见 G4）。
4. ratio 1 的 `state_cache is None` 分支只在 `test_v41_fused_save_compress_and_insert` 的 cr=1 参数下走到，但断言只查输出 latent，不查它是否真的避开了 ring。

### 10.4 最小验证命令（H200 上）

```bash
python -m pytest tests/kernels/test_compressor_kv_cache.py -q
python -m pytest tests/v1/attention/test_deepseek_v4_swa_visible.py -q
```

---

## 附录 A · 关键文件与可复核命令

| 文件 | 作用 |
|---|---|
| `vllm/models/deepseek_v41/compressor.py` | `DeepseekCompressor`、`CompressorBackend/MetadataBuilder/StateCache`、ring slot kernel |
| `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` | 池化/门控/RMSNorm/ring 保存 + 三种记录写入 kernel |
| `vllm/models/deepseek_v41/attention.py` | 并行调度、cache spec、SWA 写入、indexer 侧 |
| `vllm/models/deepseek_v41/sparse_mla.py` | 压缩 slot mapping、block size 解析 |
| `vllm/v1/kv_cache_interface.py` | spec 基类、页几何、ring spec 的调度器可见字段 |
| `vllm/v1/attention/backends/mla/compressor_utils.py` | `get_compressed_slot_mapping`（DCP/CP 感知） |

**可复核命令**（工作目录 `E:\dsv41\vllm_0.32.0\compressor`）：

```bash
python build_final.py          # 一键：事实/锚点审计 → 结构图 → 生成 → 校验
python build_doc.py            # 只重新生成本文档（会先校验 56 个行号锚点与推算数字）
python verify_final_doc.py     # 独立校验本文档的内容（125 项）
python make_compressor_diagram.py   # 重新生成并审计所有结构图
python compressor_torch_reference.py --audit   # 纯 Torch 参考实现 + 锚点审计
python verify_compressor_doc.py      # 源文档的语义校验（202 项）
```

生成链的每一环都是**先校验再写文件**：`build_doc.py` 会逐个核对它引用的行号，并重新推算页大小、per-token 字节数、测试计数；任何一项对不上就拒绝写盘。

**已知局限**（如实说明）：

1. `csrc/` 层 CUDA kernel 只核到 Python 边界，没有逐字节核。
2. 模型资产没有 `.safetensors` 分片，所以张量形状与 dtype 来自 config 加参考实现，不是读张量得到的。
3. §7.4b 的 per-token 成本是**单条记录**的净宽，不含页对齐与 block 共享带来的开销 —— 估算显存请按 §7.3 的「对齐后页」乘上页数。
