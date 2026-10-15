# DeepSeek-V4.1 Sparse-Attention Indexer（H200 / sm90）— 调研与分类

> 目标硬件：**NVIDIA H200 / sm90（Hopper, compute capability 9.0）/ CUDA**
> 代码基线：`E:\dsv41\vllm_0.32.0\vllm-v0.32.0rc0\vllm-v0.32.0rc0`（vLLM v0.32.0rc0）
> 模型基线：`E:\dsv41\dsv41_part`（DeepSeek-V4.1：40 主干层 + 3 MTP 层，`compress_ratios` 43 项）
> 可运行参考实现：`..\indexer\indexer_torch_reference.py`（纯 Torch 写的，可以单独跑，18/18 自检全过；覆盖量化往返、打分、top-k、候选 mask、分页字节布局和 r=2 的分块寻址）

**记法**：`文件:行号` 用编辑器行号。短路径的根依次为 `vllm/models/deepseek_v41/`、`vllm/v1/attention/backends/mla/`、`vllm/model_executor/layers/`、`vllm/model_executor/kernels/attention/dsa/`、`vllm/model_executor/models/`、`vllm/models/deepseek_v4/`、`vllm/utils/`、`vllm/platforms/`；其他模型族（§8.2）的短路径以 `vllm/models/<族>/` 为根。每个短名只对应一个文件；**同名文件一律带目录限定**（已知易混对：`attention.py` / `compressor.py` / `cache_utils.py` / `sparse_mla.py` / `quant_config.py` / `flashinfer_sparse.py` 的 v4 与 v41 两版，`rocm_aiter_mla_sparse.py` 与 `rocm_paged_mxfp4_indexer.py` 的 `ops/` 与 `backends/mla/` 两版，以及 `tokenspeed_mla.py` 的 `prefill/` 版）——未加限定的短名一律指 `vllm/models/deepseek_v41/` 或首个匹配的根。结构类编号 `C1…C12`（§4），`X0/X1/X2` 是图编号（§7）不是结构类。索引缓冲区中 `-1` = 「无此项」。§7 的 7 张图由 `..\indexer\_work\make_diagrams.py` 程序化生成并逐格审计，图内只有 ASCII/Latin 字形。

---

## 0. 结论速览

1. **H200 上 indexer 的 K cache 只有 FP8 一种格式**：值是 e4m3，每 128 个元素配一个 **2 的幂的 fp32 scale**，值和尺度合起来算，一个 state 占 132 B；张量本身的 dtype 恒为 `torch.uint8`（`attention.py:92-100`、`:442`）。把 `indexer_kv_dtype` 设成 `"mxfp4"`，在 sm90 上会**直接抛 `ValueError`**，设成 `bf16`/`nvfp4` 同样被拒（`mla/indexer.py:62-66`、`:90-95`），**不会悄悄降级**。

2. **H200 走的是「全上下文 dense logits + mask」这条路**：`indexer_sparse_logits` 要求 `mxfp4` + SM100 + DeepGEMM ≥ 2.8 + DeepSelect 同时成立，H200 一条都不占（`config/attention.py:94-104`、`mla/sparse_indexer.py:105-153`）。⚠️ 这里有个坑：H200 上**最先撞到**的是 mxfp4 那道拒绝，而不是那句「requires an SM100-class GPU」，照着报错去查容易走偏。

3. **同一个 indexer 身上有三种 dtype 并存**：K cache 是 FP8 载荷装在 `uint8` 容器里；`wq_b` 是 **MXFP8**（e4m3 + ue8m0，block 为 `(32,32)`，`deepseek_v41/quant_config.py:240-243`）；而 `wk` / `weights_proj` / `k_norm` 是**明确不量化的 bf16**（`attention.py:1325`、`:1351`）。

4. **打分的 ReLU 在 kernel 内部做，而且是按 head 做的**：`logits[t,s] = Σ_h w[t,h]·relu(q[t,h]·k[s])`（DeepGEMM 的 `fp8_fp4_(paged_)mqa_logits`）。**如果改成「先把各 head 求和、最后再 ReLU」，选出来的 top-k 就不一样了**；两处佐证：`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:871`、`test_mla_short_prefill_indexer.py:1038`。

5. **只有低 4 层产 K，高 4 层直接复用**：`owns_k = is_kv_source`（`attention.py:472`）。L2/8/14/20 各自带一套 `wk` + `k_norm` 和自己的 K cache；L24/28/32/36 靠改写 prefix **共用 L20 的那一份**（`:449-459`）——所以全模型只有 4 个 indexer K cache 张量，这一点和检查点对得上。

6. **两层候选筛选把相邻层绑在了一起，而且有一条没人管的不变量**：L20 负责发布（`candidate_write=True`），L24/28/32/36 负责消费；缓冲区是全模型共用**的一张** `(max_num_batched_tokens, 2048)` int32，custom op 用 `mutates_args` 原地改写它（`nvidia/model.py:717-721`、`sparse_attn_indexer.py:934-940`）。候选 id 用的是**发布层的 compressed-position 坐标系**，而**代码里没有任何 assert** 把消费者的 `compress_ratio` 和发布层绑在一起——它之所以正确，只是因为出厂配置恰好两边都是 ratio 1。

7. **indexer 真正交出去的是一份索引**：`topk_indices_buffer` 是 `(max_num_batched_tokens, 512)` 的 **int32**，里面装的是 **local** compressed id，空位填 `-1`，**不保存任何 weights**；它再经 `compute_global_topk_indices_and_lens` 换算成全局 KV slot，最后交给 H200 上唯一能用的 **`FLASHMLA_SPARSE_DSV41`**。

8. **三条「看着能用、其实到不了」的路**：`mxfp4` K cache、`indexer_sparse_logits` / `SparseMQAIndexer`、DeepSelect top-k。另外注意 `has_deep_gemm_sparse_mqa()` 在 H200 上**返回 True**（它只查几个符号在不在，`utils/deep_gemm.py:759-768`），可它却被当成架构判据来用了——只是旁边的 `family(100)` 与它相与，才没出问题（§2.2）。

9. **V4.1 相对 V4 有两处结构性改动**（§8.1）：indexer 的 K 不再来自 `hidden_states` 的 GEMM，而是 `k_norm(wk(latent))`（latent 就是 kv-source compressor 输出里那 512 维的 pre-RoPE 部分）；两层候选筛选是 V4.1 才有的。另外 SWA-only 的判据从 V4 的 `<= 1` 变成了 **`== 0`**。

---

## 1. 术语、符号与数学定义

### 1.1 符号表

| 符号 | 值 | 含义 | 依据 |
|---|---|---|---|
| `T` | 动态 | 这一步真正送进 indexer 的 token 行数 | — |
| `H` | **32** | indexer head 数 = `index_n_heads` | `attention.py:1301` |
| `D` | **128** | indexer head dim = `index_head_dim` | `attention.py:1302` |
| `R` | **64** | indexer RoPE dim = `qk_rope_head_dim`（`D_nope = 64`） | `attention.py:1303` |
| `Q_LORA` | **1280** | `q_lora_rank`，`wq_b` 的输入维度 | `attention.py:1304` |
| `D_main` | **512** | 主 MLA head dim，也是 compressor latent 维度 | `attention.py:297`、`:474` |
| `K` | **512** | `index_topk` = 每 query 选出的压缩位置数 | `attention.py:1300` |
| `C_bs` / `C_topk` | **8** / **2048** | `candidate_block_size` / `candidate_topk_blocks` | `config.json:127`、`:126` |
| `r` | 1 / 2 | 本层 `compress_ratio` | §1.2 |
| `N` | `block_size // r` | 一个物理 block 存多少个压缩 state | `kv_cache_interface.py:231-233` |
| `P` | **64**（sm90） | indexer cache 的 kernel block size | `mla/indexer.py:370-372` |
| `B_row` | **132**（fp8）/ 68（mxfp4） | 每个 state 的**合计**字节宽 | `attention.py:92-100` |
| `S_page` | `round_up(N*B_row, 576)` | 物理 block stride（H200） | **两个来源要分开看**：`attention.py:1216-1218` **只设** `page_alignment=576`；真正的 `unpadded_page_size` 与 `round_up(..., alignment)` 在 `vllm/v1/kv_cache_interface.py:551-560`（属性）与 `:676-682`（`_apply_alignment_padding`，`round_up` 在 `:680`） |

### 1.2 `compress_ratios` 的三档（逐个索引核对过）

`text_config.compress_ratios` 一共 **43** 项（40 个主干层 + 3 个 MTP 层），实际布局是这样（`config.json:60-104`）：

| 索引区间 | 值 | 条目数 | 含义 | 该区间有无 indexer |
|---|---|---|---|---|
| `0..1` | **0** | 2 | 纯 sliding window | **无** |
| `2..19` | **2** | 18 | ratio-2 压缩 | 仅 L2、L8、L14 |
| `20..39` | **1** | 20 | ratio-1 压缩（full-length compressed cache） | L20、24、28、32、36 |
| `40..42` | **0** | 3 | MTP 层，纯 sliding window | **无** |

> ⚠️ 常见误写：**不是**「前 20 层 ratio 2」，也**不是**「只有最后 3 层 ratio 0」。真实分布是 `0,0` + `18×2` + `20×1` + `3×0`；**L0 与 L1 也是 ratio 0**。

代码语义：`0` = 纯 sliding window，`1` = full-length compressed，`2` = ratio-2 compressed（`attention.py:303-321`）。**ratio 本身不决定某层是否有 indexer**，决定它的是 `index_source_layer_ids` 的成员资格（`attention.py:431-432`）。`not in (0,1,2)` 抛 `ValueError`（`:316-321`）。

ratio 会牵扯到哪些地方（逐条核对过）：indexer 的 `max_model_len`/`max_total_seq_len` 都要除以 ratio（`:1334-1341`，⚠️ 若 index source 是 ratio 0 会 `// 0` 崩，**无保护**）；compressed-KV cache 只给 `is_kv_source` 层且 `tokens_per_state=r`（`:1143-1163`）；K store 的组边界与组起点（`indexer_k_store.py:153`、`:179`，`assert r in (1,2)` 在 `:61`）；metadata builder 只接受 1/2（`deepseek_v41/sparse_mla.py:172-176`）；"SWA only" 判定用 `r == 0`（`nvidia/flashmla.py:98`、`:135`、`:258`）；而「ratio 0 就不做 indexer top-k」这件事，真正处理它的是 `common/ops/cache_utils.py:1529-1530` 和 `:1087-1095`（调用方传 `TOP_K=0`，免得 constexpr 里出现 `// 0` 的 UB）。

### 1.3 indexer 打分公式（Lightning Indexer）

```
I[t, s] = Σ_{h=0..H-1} w[t, h] · relu( q[t, h] · k[s] )
  q[t,h] ∈ R^128 : FP8 反量化后的 index query（第 t 个 token 的第 h 个 head）
  k[s]   ∈ R^128 : FP8 反量化后的 index key（第 s 个 compressed position）
  w[t,h]         = weights_proj(hidden_states)[t,h] · q_scale[t,h]
                   · softmax_scale · head_scale
  softmax_scale  = 128^(-1/2) ≈ 0.0883883        (attention.py:1328)
  head_scale     =  32^(-1/2) ≈ 0.1767767        (attention.py:1460)
logits[t, s] 在 [ks[t], ke[t]) 之外置 -inf，然后按行 top-K（K = 512）。
```

`relu` 有没有、放在哪一步，有三处独立证据：`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:871`（同一对象的 ROCm 回退实现显式写 `score.relu()`）、`test_mla_short_prefill_indexer.py:1038`、`test_deepgemm_attention.py:88`。

> ⚠️ **同名文件陷阱**：仓库里有两个 `rocm_aiter_mla_sparse.py` —— `vllm/v1/attention/**ops**/` 与 `vllm/v1/attention/**backends/mla**/`。
> 上述 `score.relu()` 在 **`ops/` 那份的第 871 行**；`backends/mla/` 那份**全文零 `relu`**（它第 871 行是 sink-attention 回退逻辑）。引用时必须带 `ops/` 目录。

### 1.4 权重折叠的约定（FP8 与 MXFP4 的分水岭）

| 路径 | `q_quant` 返回值 | `weights_out` | 依据 |
|---|---|---|---|
| **FP8（H200 唯一可达）** | 单张量 `(T,H,D)` e4m3fn，**无伴随 scale** | `w · q_scale · softmax_scale · head_scale`，q_scale **被折进去**；dtype 强制 fp32 | `fused_indexer_q.py:189-207`、`:671` |
| MXFP4（H200 不可达） | 二元组 `(q_packed uint8, q_scale int32)` | `w · softmax_scale · head_scale`，**不折** q_scale | `fused_indexer_q.py:428-438`、`:666-669` |

所以 op 里有一对互斥的 assert（`sparse_attn_indexer.py:505-510`）：`use_fp4_cache` 时 `q_scale is not None`，否则 `q_scale is None`。

### 1.5 量化 scale 怎么算（两侧都是 2 的幂）

| 侧 | 规则 | 位置 |
|---|---|---|
| Q | `q_scale = 2**ceil(log2(max(amax, 1e-4) / 448))` | `fused_indexer_q.py:163-164`（Triton）/ `nvidia/ops/fused_indexer_q_cutedsl.py:541`（CuTeDSL，floor 同为 **1e-4**） |
| K（FP8） | `scale = 2**ceil(log2(max(absmax,1e-4) / 448))`，**每 128 宽行一个 fp32** | `indexer_k_store.py:241-245`、`:255` |
| K（MXFP4） | `ue8m0 = 2**ceil(log2(max(amax, 6·2^-126) / 6))`，每 32 元素一个字节 | `indexer_k_store.py:218-236` |

因为 scale 是 2 的幂，「反量化值 = scale × e4m3 网格」这一步是**精确**的（参考实现的自检 S2/S9 断言残差为 0），误差全都出在 e4m3 只有 3 位尾数上。

---

## 2. H200/sm90 运行时后端门控矩阵（最关键的一章）

### 2.1 平台判定规则（后面全靠它）

| 谓词 | 位置 | 语义 | H200 |
|---|---|---|---|
| `is_device_capability_family(cap)` | `platforms/interface.py:576-588` | **只比 major**：`to_int()//10 == cap//10` | `family(90)`=**T**、`family(100)`=**F**、`family(120)`=**F** |
| `is_device_capability(cap)` / `has_device_capability(cap)` | `:553-574` / `:530-551` | 精确相等 / `>=` | `(90)`=**T**、`(100)`=**F** |
| `is_cuda` / `is_rocm` | `:196-209` | 平台枚举 | CUDA=**T**，ROCm=**F** |
| `support_deep_gemm`（CUDA） | `platforms/cuda.py:688-695` | `capability(90) or family(100) or family(120)` | **T** |
| `has_deep_gemm()` | `utils/import_utils.py:512-519` | **只检查包能否 import，不是架构判据** | 取决于环境 |

**`family(90)` 和 `family(100)` 互斥**，所以每个 `family(100)` 门控在 H200 上都是一堵过不去的墙，跟装没装 DeepGEMM 没关系。

### 2.2 门控真值表（indexer 相关）

| 门控 / 判据 | 位置 | H200 | 效果 |
|---|---|---|---|
| `use_fp4 and not family(100)` → `ValueError` | `mla/indexer.py:90-95` | **F，抛异常** | 禁止 `indexer_kv_dtype="mxfp4"` |
| `kv_dtype not in ("fp8","mxfp4")` → `ValueError` | `mla/indexer.py:62-66` | 用户显式设才抛 | 只允许 fp8 / mxfp4 |
| `resolve_indexer_kv_dtype("fp8")`（`auto` → `fp8`） | `mla/indexer.py:61`；`config/attention.py:163-167` | 返回 `"fp8"` | **锁定 FP8** |
| `not (is_cuda() and family(100))` → `ValueError` | `mla/sparse_indexer.py:116-120` | **F，抛异常** | 禁止 sparse-logits metadata |
| sparse 路径的前置 `dsa_indexer_uses_fp4()` | `mla/sparse_indexer.py:110-115` | **F，先抛** | 禁止 sparse-logits（**早于** SM100 报错） |
| `not (is_cuda() and family(100) and has_deep_gemm_sparse_mqa())` | `sparse_mqa_indexer.py:99-106` | **F，抛异常** | 禁止 `SparseMQAIndexer` |
| `has_deep_select()` = spec ∧ `is_cuda()` ∧ `family(100)` | `kernels/attention/dsa/sparse_mqa_logits.py:43-49` | **F** | 无 DeepSelect top-k |
| `has_deep_gemm_sparse_mqa()` = 4 个符号非空 | `utils/deep_gemm.py:759-768` | **T** ⚠️ | **不是架构判据**，却被当判据用 |
| `_supports_varlen_paged_mqa_logits()` | `mla/indexer.py:932-937` | **F** | 禁止 varlen decode 布局 |
| `_supports_flattened_device_query_lens()` = `is_cuda() ∧ family(90) ∧ has_deep_gemm()` | `mla/indexer.py:940-945` | **T**（**但依赖 `has_deep_gemm()`**：DeepGEMM 包缺失时为 **F**） | Hopper 的摊平替代路径 |
| `native_next_n_supported(next_n)` = `next_n in (1,2,4)` | `utils/deep_gemm.py:640-648` | **T for 1/2/4** | native spec-decode 合法性 |
| `_cooperative_capable` = `has_device_capability(90) and not family(120)` | `indexer_topk.py:160-163` | **T** | cooperative 成为 decode top-k 默认 |
| `_has_deep_select` = `is_cuda() and family(100)` | `indexer_topk.py:156-158` | **F** | 显式 `deep_select` 会 RuntimeError |
| `[64 if family(90) else 128]`（indexer / 主 MLA） | `mla/indexer.py:370-372`；`deepseek_v41/sparse_mla.py:88-90` | **`[64]`** | indexer cache 与主 KV page = 64 token |
| `_use_v41_mxfp8_kv_record()` = `family(100)` | `attention.py:128-129` | **F** | 用 584 B 的 V4 record、576 B 对齐 |
| mega-attn 需要 `family(100)` | `flash_mla_mega_attn.py:96-97` | **F** | `FLASHMLA_MEGA_ATTN_DSV41` 不可用 |
| `supports_compute_capability` major ∈ `[9,10]` | `deepseek_v41/sparse_mla.py:123-125` | **T** | H200 的 MLA 消费端合法 |
| `FLASHINFER_MLA_SPARSE_DSV41` 需要 major ∈ `{10,12}` | `deepseek_v41/nvidia/flashinfer_sparse.py:136-138` | **F** | 该后端不可用 |
| `is_cuda() and not has_deep_gemm()` → `RuntimeError` | `sparse_attn_indexer.py:1003-1007` | 需 DeepGEMM | CUDA indexer op 的硬前提 |
| `if has_cutedsl(): _INDEXER_Q_FP8_KERNEL` | `fused_indexer_q.py:676-691` | **取决于环境** | 装了 `cutlass` 走 CuTeDSL，否则走 Triton |
| `indexer_sparse_logits: bool = False` | `config/attention.py:94` | **F（默认）** | 主开关默认关 |

`has_cutedsl()` 就是 `_has_module("cutlass")`（`import_utils.py:647-649`），**不带任何架构门控**；所以 indexer 的 Q kernel 在 H200 上走哪条分支，完全看环境，两条分支实现的是同一套算术。

### 2.3 走不通的路（分三类，别混）

| 项 | 拒绝点 | 类型 |
|---|---|---|
| `indexer_kv_dtype="mxfp4"` on sm90 | `mla/indexer.py:90-95` `ValueError` | 抛异常 |
| `indexer_kv_dtype` ∈ {`bf16`,`nvfp4`} | `mla/indexer.py:62-66` `ValueError` | 抛异常 |
| `indexer_sparse_logits=True` on sm90 | `mla/sparse_indexer.py:110-115` → `:116-120` → `:121-125` → `:126-129` → `:130-133` → `:137-141` → `:142-147`（连着七道拒绝，**第一道就是 mxfp4**） | 抛异常 |
| `SparseMQAIndexer` 构造 on sm90 | `sparse_mqa_indexer.py:99-111` `ValueError` | 抛异常 |
| 显式 `sparse_indexer_topk_backend="deep_select"` | `indexer_topk.py:192-203` → `:229-233` `RuntimeError` | 抛异常 |
| 显式 `FLASHINFER_MLA_SPARSE`（非 DSV41 名） | `nvidia/model.py:208-216` `ValueError` | 抛异常 |
| `nvfp4_ds_mla` 且 `not kv_mxfp8` | `attention.py:523-527` `ValueError` | 抛异常 |
| 候选筛选 + DCP | `sparse_attn_indexer.py:435-441` `assert` | 抛异常 |
| `wk(latent)` 查不到源 cache（PP 切在 kv-sharing group 内） | `attention.py:454-459`、`:596-601` `NotImplementedError` | 抛异常 |
| DeepSelect / `deep_select_topk`、DeepGEMM `fp8_fp4_(paged_)sparse_mqa_logits` | 被 `family(100)` 挡在前面 | 静默不可达 |
| ROCm 的 `SHUFFLE` tiled store、`forward_hip`/`forward_cpu`/XPU 分支 | `indexer_k_store.py:80-92`；`sparse_attn_indexer.py:1054-1066`、`:666-668`、`:809-811` | 静默不可达 |
| `FLASHINFER_MLA` / `CUTLASS_MLA` / `TOKENSPEED_MLA`（major 门控） | `mla/flashinfer_mla.py:219-221`；`mla/cutlass_mla.py:65-66`；`mla/tokenspeed_mla.py:125-126` | 静默不可达 |
| ROCm 专用 indexer 后端 `DeepseekV41RocmMxfp4IndexerBackend` 与层 `RocmSparseAttnIndexer` | 入口 `attention.py:1194-1196` 的 `self.rocm_mxfp4 = is_rocm() ∧ dsa_indexer_uses_fp4()` 恒为假；后端分支 `:1244-1250`，层类在 `model_executor/layers/rocm_paged_mxfp4_indexer.py:22` | 静默不可达 |
| varlen decode → 摊平成单 token 行 | `mla/indexer.py:1033` → `:1599-1606` | 静默降级 |
| mega-attn 拒绝 → `DeepseekV4FlashMLAAttention` | `nvidia/model.py:241-243` | 静默降级 |
| V4.1 MXFP8 KV record → 584 B V4 record；`block_stride_alignment` → `None` | `attention.py:1216-1218`、`:1234-1238` | 静默降级 |

### 2.4 H200 上真正会走的那条路（全都在这儿）

```
_select_dsv4_attn_cls          nvidia/model.py:197-243
  mega-attn.is_available_for = False（family(100)）→ DeepseekV4FlashMLAAttention
  backend_cls = DeepseekV4FlashMLABackend（"FLASHMLA_SPARSE_DSV41"）  nvidia/flashmla.py:53
→ DeepseekV4Attention.__init__  attention.py:274    ratio / is_kv_source / is_index_source
  indexer（8 层）  :431-482        index_k_cache（4 层）:437-446
  use_fp4_kv=False :1307            B_row=132 / page_alignment=576 :92-100 / :1216-1218
  backend = DeepseekV41IndexerBackend，block sizes [64]   :1243-1252 / mla/indexer.py:370-372
→ indexer_op = SparseAttnIndexer（use_sparse_logits=False）  attention.py:1383-1398
→ Q：fused_indexer_q_rope_quant(use_fp4=False, weights_out_dtype=fp32)
     has_cutedsl() ? IndexerQFp8Kernel : FusedIndexerQRopeQuantTritonKernel
→ K：indexer_k_norm_rope_store(USE_FP4=False, SHUFFLE=False)
→ forward()：max_seq_len // r <= 512 且非 capture → C9 快路径直接返回  :1482-1504
     否则 _produce_k → forward_q → sparse_attn_indexer
→ 打分：prefill fp8_fp4_mqa_logits + top_k_per_row_prefill
        decode  fp8_fp4_paged_mqa_logits + get_indexer_topk
        候选    select_candidate_blocks(L20) / apply_candidate_mask(L24+)
→ decode top-k auto：cooperative(rows≤64) → persistent → per_row
→ metadata builder：use_flattening=False for next_n∈{1,2,4}；block 64；DCP/PCP/TP 分片
→ 消费：compute_global_topk_indices_and_lens → combine_topk_swa_indices → flash_mla_sparse_fwd
```

参考实现的自检 S17 把这段格式锁定的算术逐条断言了一遍（`family(100)=False` → 132 B 行、576 B 对齐、8640 B page）。

---

## 3. 注册的算子清单

### 3.1 核心实现文件（行数就是复杂度的信号）

| 文件 | 行数 | 角色 |
|---|---|---|
| `mla/indexer.py` | 1972 | 后端主实现：门控、backend 类、metadata builder、TP row-shard、DCP/PCP |
| `sparse_attn_indexer.py` | 1358 | `sparse_attn_indexer` custom op + `SparseAttnIndexer` 层 |
| `attention.py`（v41） | 1522 | 模型集成：`DeepseekV4Indexer`、`DeepseekV4IndexerCache`、跨层接线 |
| `fused_indexer_q.py` + `fused_indexer_q_cutedsl.py`（v4 树，V4.1 复用） | 733 + 708 | Q 侧融合 kernel 的 Triton / CuTeDSL 两个变体 |
| `indexer_topk.py` | 405 | decode top-k 后端选择与执行 |
| `indexer_k_store.py`（v41 新增） | 255 | K 侧融合 kernel |
| **`common/ops/cache_utils.py`（v41）** | **1587** | **上一版漏了这一行**——C10 的实现全在这里（slot mapping、`combine_topk_swa_indices`、topk lens、候选块工具）；**它是 indexer 里第二大的文件**，只比 `mla/indexer.py` 小 |
| **`compressor.py`（v41）** | **327** | `D_main` 的生产者（compressor 本体，详见 compressor 子系统的文档） |
| **`sparse_mla.py`（v41）** | **283** | C12 后端 `DeepseekV4SparseMLABackend` / metadata builder |
| **`nvidia/flashmla.py`（v41）** | **359** | C12 消费端（FlashMLA sparse decode / prefill 的实际调用点） |
| `candidate_blocks.py` | 235 | 两层候选筛选的发布 / 消费 kernel |
| `sparse_mqa_indexer.py` + `mla/sparse_indexer.py` | 209 + 235 | candidate-sparse 打分层与后端（**H200 不可达**）；`config/attention.py`（227）管配置 |

### 3.2 逐个算子看（H200 视角）

| 算子 / kernel | 位置 | H200 |
|---|---|---|
| `fused_indexer_q_rope_quant`（FP8 分支） | `fused_indexer_q.py:549`（分派）· `:717` Triton · `:682` CuTeDSL | ✅ Q 侧唯一路径 |
| `FusedIndexerQRopeMxFp4TritonKernel` | `fused_indexer_q.py:322` | ❌ |
| `indexer_k_norm_rope_store` | `indexer_k_store.py:29`（入口）· `:122`（kernel） | ✅ K 侧唯一路径 |
| `sparse_attn_indexer`（custom op） | `sparse_attn_indexer.py:400`·注册 `:934-940` | ✅ 打分层 |
| `ops.cp_gather_indexer_k_quant_cache` | `csrc/cache_kernels.cu:687` | ✅ prefill K gather |
| `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` | `utils/deep_gemm.py:594`（def）/ `:694` | ✅ prefill / decode 打分 |
| `ops.top_k_per_row_prefill` | `_custom_ops.py` | ✅ prefill top-k |
| `torch.ops._C.cooperative_topk` / `persistent_topk` / `ops.top_k_per_row_decode` | `indexer_topk.py:322` / `:339` / `:396` | ✅ decode top-k 三级链 |
| `_select_candidate_blocks` / `_apply_candidate_mask` | `candidate_blocks.py:141` / `:192`（共 5 个 Triton kernel） | ✅ 候选发布 / 消费 |
| `_fill_short_context_topk_indices` | `attention.py:103`（kernel）· `:1496`（调用） | ✅ 短上下文快路径 |
| `compute_global_topk_indices_and_lens` / `combine_topk_swa_indices` | `common/ops/cache_utils.py:832` / `:925` | ✅ 索引 → 全局槽位 / SWA 合并 |
| `_gather_workspace_shapes` / `kv_cache_as_quant_view` | `sparse_attn_indexer.py:289` / `:310` | ✅ decode 视图与工作区 |
| `PrefillRowShard.exchange_topk`（PyNCCL all_gatherv） | `sparse_attn_indexer.py:360` | ✅ 仅 TP>1 且行数超阈值 |
| `maybe_gather_indexer_k` / `dcp_gather_kv_rows` / `_merge_dcp_topk_global` | `v1/attention/ops/pcp.py`；`sparse_attn_indexer.py:146` / `:93` | ✅ 仅 PCP / DCP |
| `flash_mla_sparse_fwd` | `v1/attention/ops/flashmla.py` | ✅ MLA 消费端 |
| `torch.ops.deep_select.topk` | `indexer_topk.py:111` | ❌ family(100) |
| `fp8_fp4_(paged_)sparse_mqa_logits` | `utils/deep_gemm.py:759-768` 探测 | ❌ SM100-only |
| ROCm aiter / tiled store / FNUZ；XPU / CPU forward | `sparse_attn_indexer.py:1120-1165`、`indexer_k_store.py:80-92`、`:666-676`、`:1167-1358` | ❌ 平台不符 |

> ⚠️ **同名冲突**：仓库里有**两个** `fused_indexer_q_rope_quant`。上表那个（`deepseek_v4/common/ops/fused_indexer_q.py:549`）才是 V4.1 用的；另一个在 `sparse_attn_indexer.py:245`，是 FP8-only 的 Triton 版（多一个 `is_neox` 参数），只有 DeepSeek-V3.2 在用（`model_executor/models/deepseek_v2.py:85-88`、`:797`）。两者约定不同：前者支持 MXFP4，也支持 GPT-J 交错 RoPE。

---

## 4. indexer 类型分类（12 类）

编号 `C1..C12` 全书和结构图都用它。**这 12 类在 H200 上都会跑到**（个别类里含有走不到的分支，已在类内标注）。
回头对一遍：§2.4 那条路上的每一步都能归到某个编号，没有遗漏；各类的调用关系和位置见 §7 图 X0，逐类说明在下面各小节。

依次为：`C1` 容器与权重 · `C2` Q 侧融合 kernel · `C3` K 侧融合 kernel · `C4` index K cache 与分页几何 ·
`C5` dense 打分 · `C6` decode top-k 后端选择 · `C7` prefill top-k · `C8` 两层候选筛选 ·
`C9` 短上下文快路径 · `C10` 索引→全局槽位 + SWA 合并 · `C11` 后端 / metadata / 并行分片 · `C12` MLA 消费端。

---

### C1 — indexer 容器与权重（`DeepseekV4Indexer`）

- **位置 / 代码**：8 个 index-source 层；`attention.py:1255-1522`（层）· `:1314-1354`（权重）
- **作用**：管着 `wq_b` 和 `weights_proj`；本层要产 K 时（`owns_k`）还多一组 `wk` + `k_norm`。
- **输入 → 输出**：`qr (T,1280) bf16` → `index_q (T,32,128) bf16`；`hidden_states (T,5120) bf16` → `indexer_weights (T,32) bf16`；`latent (T,512) bf16` → `k_pre (T,128) bf16`（仅 `owns_k`）。
- **格式**：`wq_b` = **MXFP8**（e4m3 + ue8m0，block `(32,32)`）；`.scale` on-disk `(128,40)` uint8 → in-memory `wq_b.weight_scale (4096,40)`（`nvidia/model.py:1429-1460` + `modelopt.py:2245-2259`）；`wk`/`weights_proj`/`k_norm` = **bf16 不量化**。
- **触发 / sm90**：只要有 `is_index_source` 就建（`attention.py:432`），K 侧还要再看 `owns_k = is_kv_source`（`:472`）。几个常量：`softmax_scale=128**-0.5`（`:1328`）、`scale_fmt="ue8m0"`（`:1330`）、`quant_block_size=128`（`:1331`）——最后这个和检查点里的 `weight_block_size=[32,32]` **不是一回事，别混**。

### C2 — Q 侧融合 kernel（`fused_indexer_q_rope_quant`，FP8）

- **位置 / 代码**：`indexer.forward_q`（`attention.py:1439-1468`）；`fused_indexer_q.py:549-729`（分派 `:676-691`）
- **作用**：做 GPT-J 交错 RoPE（只动每个 head 的**后 64 维**），再做 per-token-per-head 的 FP8 量化，最后把 `q_scale`/`softmax_scale`/`head_scale` 一起折进 per-head 的 weights。
- **输入 → 输出**：`positions (T,) int64`、`index_q (T,32,128) bf16`、`cos_sin_cache (max_pos,64) fp32`、`index_weights (T,32) bf16` → `(q_fp8 (T,32,128) e4m3fn, weights_out (T,32) fp32)`；**无独立 q_scale 张量**。
- **格式**：e4m3fn（OCP，非 FNUZ）；`q_scale` 是 2 的幂（`:163-164`）；RoPE 输出做一次 `fp32→bf16→fp32` 往返（`:155-156`）。
- **触发 / sm90**：只要没命中 C9 就会走这里；FP8 分支强制要求 `weights_out_dtype == float32`（`:671`）。实际跑的是 `has_cutedsl()` ? `IndexerQFp8Kernel`(CuTeDSL) : `FusedIndexerQRopeQuantTritonKernel`。**`use_fp4=True` 那条分支到不了**。

### C3 — K 侧融合 kernel（`indexer_k_norm_rope_store`，FP8）

- **位置 / 代码**：仅 4 个 kv-source 层（L2/8/14/20）的 `indexer._produce_k`（`attention.py:1405-1437`）；`indexer_k_store.py:29-119`（入口）· `:122-255`（kernel）
- **作用**：`latent → wk → k_norm → RoPE@组起点 → FP8 → 分页写入`，**原地写入、无返回值**。
- **输入 → 输出**：`k_pre (T,128) bf16`、`positions (T,) int64`、`cos_sin (max_pos,64) fp32`、`k_norm.w (128,)`、`k_cache [nb, P/r, 132] uint8`、`slot_mapping (T,)`、`r∈{1,2}`、`use_fp4=False` → 无返回。
- **格式**：每 128 宽行**一个** 2 的幂 fp32 scale（`FP8_MAX=448`，absmax 下限 `1e-4`）；内部 assert `k_pre` 形状/dtype/stride 与 `r in (1,2)`（`:57-61`）。
- **触发 / sm90**：`(pos+1) % r == 0`（**组边界**，`:153`）且 `slot >= 0`；RoPE 位置取**组起点** `(pos//r)*r`（`:179`）。实际跑的是 Triton 的 `_indexer_k_norm_rope_quant_store_kernel`，`TOKEN_STRIDE=128`、`SCALE_DIM=4`、`SHUFFLE=False`、`USE_FP4=False`。

### C4 — index K cache 对象与分页几何（`DeepseekV4IndexerCache`）

- **位置 / 代码**：每层一个对象，只有 kv-source 层绑定并写入（`attention.py:437-446`），其余复用 L20 的对象（`:449-459`）；对象 `:1175-1252`，spec `:1206-1239`
- **作用**：声明 indexer cache 的 dtype / page 对齐 / block stride 对齐，并按配置选后端。
- **输入 → 输出**：`head_dim=132` → `MLAAttentionSpec(head_size=132, dtype=uint8, tokens_per_state=r, alignment=576)`；绑定后 `[num_blocks, P//r, 132] uint8`。
- **格式**：容器恒 `torch.uint8`，载荷 FP8；`page_alignment=576`（`:1216-1218`）；`block_stride_alignment=None`（`:1234-1238`）。
- **触发 / sm90**：`is_index_source` 建对象；`get_attn_backend()` 的三路选择（`attention.py:1243-1252`）在 H200 落 **`DeepseekV41IndexerBackend`**（`self.rocm_mxfp4` 与 `self.sparse_logits` 均为假），block size `[64]`（`mla/indexer.py:370-372`）。
  三路全貌（上一版只写了走哪一路，另两路没点名）：
  ```python
  if self.rocm_mxfp4:      return DeepseekV41RocmMxfp4IndexerBackend   # v1/attention/backends/mla/rocm_paged_mxfp4_indexer.py
  if self.sparse_logits:   return DeepseekV41SparseIndexerBackend     # 候选块打分（sm90 默认关）
  return DeepseekV41IndexerBackend                                    # ← H200 走这条
  ```
  > 另注：**层实现类** `RocmSparseAttnIndexer`（`SparseAttnIndexer` 的子类）在 `vllm/model_executor/layers/rocm_paged_mxfp4_indexer.py:22`，与上面那个 **backend 类**不同文件；两者已在 §2.3 与 §5.3 中列出。
- ⚠️ `:1334-1341` 会算 `max_model_len // compress_ratio`：如果有哪一层是 index source、`compress_ratio` 又是 0，这里就会 `// 0` 崩掉（**没有保护**）。

### C5 — dense 打分（prefill / decode）

- **位置 / 代码**：`_sparse_indexer_and_attn`（`attention.py:985-1009`）→ `indexer.indexer_op(...)`；`sparse_attn_indexer.py:400-899`（prefill `:652-720`，decode `:739-897`）
- **作用**：算 `logits[t,s] = Σ_h w[t,h]·relu(q[t,h]·k[s])` 的**全上下文** causal-masked 版本。
- **输入 → 输出**
  | 分支 | 输入 | 输出 |
  |---|---|---|
  | prefill | `q (rows,32,128)` e4m3fn；`k_quant (S,128)` fp8 + `k_scale (S,4)` uint8；`weights (rows,32)` fp32；`cu_seqlen_ks/ke` int32 | `logits (rows,width)` fp32，`[ks,ke)` 外 `-inf` |
  | decode | `q (B,next_n,32,128)`；`kv_cache [nb,P,1,132]` 视图；`weights (B*next_n,32)`；`seq_lens (B,next_n)`；`block_table`；`schedule_metadata` | `logits (B*next_n, max_model_len//r)` fp32 |
- **格式**：logits 恒 **fp32 dense**（`clean_logits=False`，`:684`、`:833`）；H200 上根本没有「只对候选块算 sparse logits」这条路。FP8 的那对 assert 在 `:505-510`。
- **触发 / sm90**：实际走的是 prefill 的 `fp8_fp4_mqa_logits`（`:678-685`，先用 `cp_gather_indexer_k_quant_cache` 把 K 拼成连续内存）和 decode 的 `fp8_fp4_paged_mqa_logits`（`:825-835`）。**`supports_varlen=False`**；decode 借 `_supports_flattened_device_query_lens()` 把每个 query 摊平成单 token 行。

### C6 — decode top-k 后端选择（`SparseIndexerTopk`）

- **位置 / 代码**：`sparse_attn_indexer.py:869-876` → `indexer_topk.py:141-405`（`auto` 链 `:236-253`，约束 `:255-276`）
- **作用**：在 decode 的 logits 上逐行取 top-512；`auto` 的优先顺序是 **cooperative → persistent → per_row**。
- **输入 → 输出**：`logits (rows, max_model_len//r) fp32`、`seq_lens`、`next_n`、`topk_tokens=512`、`max_seq_len` → 原地写 `topk_indices (rows,512) int32`，`-1` 填充。
- **格式 / 触发**：纯索引通路 int32；`RADIX_TOPK_WORKSPACE_SIZE = 1 MiB`（`:15`）；cooperative 需 `num_rows ≤ 64`（`:21`）、`topk ∈ {512,1024,2048}`、`stride(0) % 4 == 0`、`_cooperative_capable`（H200 为真）。
- **sm90**：主要走 `cooperative_topk`（`:322`），条件不满足就回落到 `persistent_topk`（`:339`）、`ops.top_k_per_row_decode`（`:396`）。`deep_select` 在 H200 上用不了；显式指定一个不满足条件的后端会直接 `RuntimeError`（`:229-233`），**不会悄悄降级**。
- ⚠️ **两个在 H200 上真的能用、但绕过了 `auto` 链的显式后端**（上一版漏了）：
  | 后端 | 接受条件（`indexer_topk.py:217-228`） | H200 上是否满足 |
  |---|---|---|
  | `sparse_indexer_topk_backend="flashinfer"` | `is_cuda()` ∧ `self._has_flashinfer_topk = has_flashinfer()`（`:159`；真正的 `flashinfer.topk.top_k_ragged_transform` 在 `:349` 惰性导入）∧ **logits 为 fp32 且 `stride(1)==1`** | ✅ **H200 的 dense-logits 路径恰好就是 fp32 + 连续** → 满足即走 flashinfer，**完全绕过 cooperative/persistent/per_row 三级链**（执行体 `:347-358`） |
  | `sparse_indexer_topk_backend="torch"` | 仅需 CUDA（无额外条件） | ✅ 可任意使用（**调试参考实现**，mask 后 `topk`，`:359-371`） |
  | `"deep_select"` | 需 CUDA ∧ `family(100)` | ❌ sm90 不可达 |
  | `"aiter"` | 需 ROCm | ❌ |
  > 这两个后端**不带任何架构门控**，所以「H200 的 decode top-k 只会走 cooperative→persistent→per_row」这句话，只对 **`auto`** 成立。

### C7 — prefill top-k

- **位置 / 代码**：`sparse_attn_indexer.py:711-720`
- **作用**：在**已含因果 mask + 候选 mask** 的 logits 上每行取 top-512，原地写 int32、`-1` 填充。
- **顺序不可交换**：候选发布读的是**未 mask 的块内最大值**，所以 C8 必须插在 logits 与 top-k 之间（`:687-710`）。

### C8 — 两层候选筛选（`candidate_blocks.py`）

- **位置 / 代码**：L20 发布（`candidate_write=True`），L24/28/32/36 消费；调用点 prefill `:694-710`、decode `:846-864`。实现 `candidate_blocks.py:141-189`（发布，3 kernel）· `:192-235`（消费，2 kernel）
- **作用**：发布 = 块内最大 logit 的 top-2048 块 id；消费 = 把「非因果 或 非候选」的列置 `-inf`。
- **输入 → 输出**：`logits (rows,width) fp32`、`starts/ends`、`topk_blocks=2048`、`block_size=8` → 发布写 `candidate_blocks (rows,2048) int32`；消费原地改 `logits`。
- **格式 / 语义**：块分数 = 块内**最大值**且 **NaN 传播**（`:9-11`、`:45`）；每行**最新块被 `+inf` 钉住**（`:46-50`）；只有 `-inf` 是 padding（NaN 分数会被保留，`:69`）；flags 宽 `nblocks+1`，越界候选 clamp 到哨兵槽 `nblocks`（`:103`），mask kernel 读回它当 `edge`（`:132-133`）。
- **触发**：`candidate_blocks is not None`；DCP 下被 `assert` 拒绝（`sparse_attn_indexer.py:435-441`）。
- ⚠️ store mask 是 `(cols < width) & ~(valid & keep)`（`:134-138`），所以「保留最后一列」**只在最后一列因果有效时**成立。

### C9 — 短上下文快路径

- **位置 / 代码**：kernel `attention.py:103-119`，调用 `:1482-1504`
- **作用**：候选数 ≤ topk 时全选（`0..num_compressed-1` 然后 `-1`），跳过整个 indexer op；**命中时仍会调 `_produce_k`**（注释 `:1486-1487`）然后返回 `(None, None, None)`。
- **输入 → 输出**：`positions (T,) int64`、`TOP_K=512`、`COMPRESS_RATIO=r` → 原地写 `(T,512) int32`。
- **触发**：`max_seq_len // r <= 512` 且非 graph capture。
- **与 V4 差异**：这个 kernel **和 V4 共用**——V4 里同一个 kernel 在 `deepseek_v4/attention.py:74-90`，两边**逐字节相同**，guard 在 `:1086-1088`；区别只在命中之后调什么（V4 调 `compressor(...)`）。

### C10 — 索引 → 全局槽位映射 + SWA 合并

- **位置 / 代码**：`common/ops/cache_utils.py:832-863`（映射）· `:925-971`（合并）
- **作用 / 输出**：local compressed id + `block_table` → `global_topk_indices (T,512)` int32 + `topk_lens (T,)`；再与 SWA 窗口合并成 `combined_indices` / `combined_lens`。
- **格式 / 关键点**：`slot = block_number * block_size + offset`（`:904`），`is_valid = local_idx >= 0`（`:895`）；`combined_topk` 会向上对齐到 128 的倍数（`:940-944`），这是为了满足 FlashMLA sparse prefill 对 `topk % B_TOPK == 0` 的要求。要留神：这里映射用的 `block_size` 是**主 KV cache** 的，不是 indexer 的 `P`（`nvidia/flashmla.py:186`、`:311`）。

### C11 — 后端 / metadata / 并行分片

- **位置 / 代码**：backend `mla/indexer.py:359-372`；builder `:984-1836`；TP row-shard `:208-262` + `sparse_attn_indexer.py:330-367`
- **作用**：声明后端名与 block size，构建 `DeepseekV32IndexerMetadata`（`slot_mapping` / `prefill.chunks` / `decode`），实现 TP row-shard、DCP、PCP、spec-decode 展平。
- **格式**：元数据 + int32/int64 索引张量。
- **触发 / 关键点**：`get_name() = "DEEPSEEK_V41_INDEXER"`，block size `[64]`；H200 上 cudagraph 支持是 `UNIFORM_BATCH`（`:990-998`）；DCP merge 要求 CuTeDSL（`:76-79`）。
  **TP row-shard 的完整门槛（上一版只列了 2 条，实际有 8 条）**——`tp_prefill_row_sharding_supported()`（`mla/indexer.py:245-262`）要求下面这些**全部**成立：
  ```
  is_cuda() ∧ dcp_world_size == 1 ∧ not use_pcp ∧ tp_size > 1
  ∧ not VLLM_DISABLE_PYNCCL ∧ not VLLM_USE_NCCL_SYMM_MEM
  ∧ not VLLM_BATCH_INVARIANT ∧ cudagraph_mode.mixed_mode() != FULL
  ```
  只要有一条不满足，这个优化就会**悄悄关掉**；另外还需要 `num_rows ≥ 16384 × tp_size`，而且**一旦开了候选筛选就会被禁用**（`sparse_attn_indexer.py:602-606`）。
- **配置校验**：不合法的组合在 builder 的 `__init__` 里就被挡掉（比如 DCP 配压缩 ratio 时要求 `cp_kv_cache_interleave_size % r == 0`，`:1105-1121`），也就是说**启动阶段就会失败**。

### C12 — MLA 消费端（`FLASHMLA_SPARSE_DSV41`）

- **位置 / 代码**：`attention.py:1009`（`forward_mqa`）→ `nvidia/flashmla.py:50-359`；backend 名 `deepseek_v41/sparse_mla.py:225-232`
- **作用**：按 top-k 索引做稀疏 MLA——gather KV、注意力、逆 RoPE、`wo_a`/`wo_b`。
- **输入 → 输出**：`q (T, padded_heads, 512)` bf16（`padded_heads ∈ {64,128}`，`nvidia/flashmla.py:66-74`）、主 KV cache、`topk_indices_buffer` → `attn_out (T, padded_heads, 512)` → `o_proj`。
- **格式**：主 KV record **584 B/token**、page 对齐 **576**（`attention.py:513`、`:522`）。
- **sm90 走法**：`swa_only = self.compress_ratio == 0`（`:98`、`:135`、`:258`）；decode 会先把 local id 换成全局 slot，再 `view(num_decode_tokens, 1, -1)`（`:188-195`）。**mega-attn 要 family(100)、FlashInfer sparse 要 major ∈ {10,12}，两者都用不了**，所以只能落 FlashMLA。

---

## 5. 各模块在哪、干什么（V4.1 逐层视角）

### 5.1 逐层矩阵

| 层 | `ratio` | indexer | `wk`/`k_norm` | index K cache | compressor | 候选筛选里的角色 |
|---|---|---|---|---|---|---|
| 0, 1 | **0** | **无** | — | — | 无 | — |
| **2** | 2 | 有 | **有** | **自有** | 有 | `buffer=None`（纯 dense） |
| 3–7, 9–13, 15–19 | 2 | **无** | — | — | 无 | — |
| **8** | 2 | 有 | **有** | **自有** | 有 | `buffer=None` |
| **14** | 2 | 有 | **有** | **自有** | 有 | `buffer=None` |
| **20** | 1 | 有 | **有** | **自有** | 有 | **发布者** `candidate_write=True` |
| 21–23, 25–27, 29–31, 33–35, 37–39 | 1 | **无** | — | — | 无 | — |
| **24 / 28 / 32 / 36** | 1 | 有 | **无** | **共享 L20** | 无 | 消费者 |
| `mtp.0/1/2` | **0** | **无** | — | — | 无 | — |

依据：`attention.py:310-334`、`:431-482`、`:576-585`；`config.json:60-127`。`nvidia/model.py:710-712` 的注释里也写着「later ratio-1 indexers (24/28/32/36)」，与此一致。

### 5.2 执行链，以及和 compressor 咬合的地方

```
DeepseekV4DecoderLayer.forward                nvidia/model.py:283+  (attn = _select_dsv4_attn_cls)
  └─ DeepseekV4Attention.forward               attention.py:659-687
       ├─ _run_parallel_input_projections      attention.py:932-983
       │    ├─ fused_wqa_wkv（默认流）           :920-921
       │    ├─ compressor_kv_score（aux0）       :954-961
       │    └─ indexer.weights_proj（aux1）      :966-971      ← indexer 在此进入
       ├─ _split_qkv_and_norm                  attention.py:773-802
       └─ _prepare_and_attn[_eager]            attention.py:804-914
            ├─ project_query_and_cache_kv  ∥  compressor(kv_score, positions)   :868-879
            ├─ prepare_indexer             ∥  compressor.insert_cache(...)      :893-902
            │    └─ DeepseekV4Indexer.forward  attention.py:1470-1511
            │         ├─（短上下文）_produce_k + _fill_short_context_topk_indices :1482-1504
            │         └─ _produce_k(:1405-1437) → forward_q(:1439-1468)
            └─ _sparse_indexer_and_attn     attention.py:985-1009
                 ├─ indexer.indexer_op(...) → sparse_attn_indexer custom op
                 └─ forward_mqa(...)        → FLASHMLA_SPARSE_DSV41
```

**咬合的地方**：indexer 和 compressor 共用 `execute_in_parallel` / `maybe_execute_in_parallel` 这条并行链（`attention.py:871-877`、`:893-900`），而且 **indexer 的 K 直接吃 compressor 吐出来的 latent**（`attention.py:714-715`：`self.indexer._produce_k(latent, positions, self.indexer_rotary_emb)`）。这正是 §8.1 里最要紧的结构差异。要留神：**`weights_proj` 的输入是 `hidden_states`，`wk` 的输入是 compressor 的 latent**——两个**不同的来源**，别当成一回事。这个并行开关的条件是 token 数 `<= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`（`:979-980`）。

### 5.3 哪些模块明确不涉及 indexer

| 模块 / 层 | 原因 | 依据 |
|---|---|---|
| L0、L1 与 3–7 / 9–13 / 15–19 / 21–23 / 25–27 / 29–31 / 33–35 / 37–39 | 不在 `index_source_layer_ids` 里 | `config.json:112-121`；`attention.py:431-432` |
| `mtp.0/1/2`（层号 40/41/42） | `is_backbone = layer_id < num_hidden_layers(40)` 为假 | `attention.py:332`；检查点 0 个 `mtp.*` indexer key |
| vision tower（32 层） | 与文本 indexer 无任何共用符号 | `config.json:157-169` |
| MoE / FFN 路径 | indexer 完全在 attention 子模块内 | `nvidia/model.py:333-338` |
| `DeepseekV4IndexerCache` 自身 | 是 cache 容器，`forward()` 为空 | `attention.py:1241` |
| compressor / SWA cache | 独立对象，只在 4 个 kv-source 层存在 | `attention.py:575-585`、`:557-568` |
| engram（L1、L14） | 与 indexer 无交集（L1 无 indexer；L14 两者都有但互不调用） | `config.json:131-134` |
| V4 树里的 indexer 专用 kernel（`compress_norm_rope_store_triton`、`_SAVE_PARTIAL_STATES_KERNEL`） | V4.1 **完全不用** | §8.1 |
| `kimi_k3` | 显式 `indexer=None` | `kimi_k3/nvidia/mla.py:355` |
| ROCm 专用 indexer 后端 / 层（`mla/rocm_paged_mxfp4_indexer.py`、`mla/rocm_aiter_mla_sparse.py`、`model_executor/layers/rocm_paged_mxfp4_indexer.py`） | `is_rocm()` 为假；`attention.py:1194-1196` 的 `rocm_mxfp4` 恒为假 | `attention.py:1244-1250` |

---

## 6. Shape、Dtype 与格式总表

### 6.1 数值这条通路

| 张量 | shape | dtype | 量化 / 格式 | 位置 |
|---|---|---|---|---|
| `hidden_states` | `(T, 5120)` | bf16 | — | 输入 |
| `qr` | `(T, 1280)` | bf16 或 MXFP8 `QuantizedActivation` | 可选融合量化（`query_quant.can_fuse_query_quant`） | `attention.py:763-802` |
| `latent` | `(T, 512)` | bf16 | 无 | `deepseek_v41/compressor.py:277-295` |
| `k_pre = wk(latent)` | `(T, 128)` | **bf16** | 显式不量化 | `attention.py:1426` |
| `index_q = wq_b(qr)` | `(T, 32, 128)` | bf16 | MXFP8（权重 e4m3+ue8m0 + 动态激活） | `attention.py:1452-1453` |
| `q_fp8` | `(T, 32, 128)` | **float8_e4m3fn** | 2 的幂 per-token-per-head scale，**折进 weights** | `fused_indexer_q.py:675` |
| `weights_out` | `(T, 32)` | **float32** | 已含 q_scale | `fused_indexer_q.py:600` |
| index K cache | `[nb, P//r, 132]` | **uint8** | 载荷 e4m3 + 1 个 fp32 幂次 scale / 128 值 | `attention.py:442` |
| `k_quant` / `k_scale` 工作区（prefill） | `(S, 128)` / `(S, 4)` | fp8（e4m3fn）/ uint8 | gather 出来的连续副本 | `sparse_attn_indexer.py:305-306` |
| `logits` | prefill `(rows,width)` / decode `(B·next_n, max_model_len//r)` | **float32** | dense + mask（`clean_logits=False`） | `sparse_attn_indexer.py:678`、`:825` |
| 主 KV record（C12 消费） | 584 B/token | uint8（`fp8_ds_mla`） | `kv_mxfp8=False` | `attention.py:513`、`:522` |

### 6.2 索引这条通路（不谈精度，但正确性全靠它）

| 张量 | shape | dtype | 语义 | 位置 |
|---|---|---|---|---|
| `positions` | `(T,)` | **int64** | token 位置（kernel 硬要求） | `attention.py:1052` |
| `slot_mapping`（indexer） | `(T,)` | int32/int64 | indexer cache 槽位，`-1` 跳过 | `indexer_k_store.py:148-150` |
| `topk_indices_buffer` | `(max_num_batched_tokens, 512)` | **int32** | local compressed-position id，`-1` 填充 | `nvidia/model.py:704-708` |
| `candidate_block_buffer` | `(max_num_batched_tokens, 2048)` | **int32** | 块 id，`-1` 填充 | `nvidia/model.py:717-721` |
| `flags`（C8 内部） | `(rows, nblocks+1)` | uint8 | 候选标志 + `edge` 哨兵 | `candidate_blocks.py:206` |
| `global_topk_indices` / `topk_lens` | `(T, 512)` / `(T,)` | int32 | 全局 KV slot / 有效计数 | `common/ops/cache_utils.py:847-848` |
| `decode_indices` / `token_to_req_indices` | `(T,)` | int32 | 行 → request | `mla/indexer.py:1072-1076` |

### 6.3 ★ 格式怎么选

| `indexer_kv_dtype` | `indexer_sparse_logits` | 平台 | Q | K cache | weights | logits | 结果 |
|---|---|---|---|---|---|---|---|
| `auto` / `fp8` | `False` | **H200** | `(T,32,128)` **e4m3fn**，无伴随 scale | **132 B/state**：128 e4m3 + 1 fp32（幂次），双平面，容器 uint8 | `(T,32)` **fp32**，含 q_scale | `(rows,width)` **fp32** dense + mask | ✅ **H200 默认路径** |
| `auto` / `fp8` | `True` | 任意 | — | — | — | — | ❌ `ValueError`（`attention.py:1366-1370`） |
| `mxfp4` | `False` | sm90 | — | — | — | — | ❌ **抛 ValueError**（`mla/indexer.py:90-95`） |
| `mxfp4` | `False` | sm100 | `(T,32,64)` uint8 + `(T,32)` int32 scale | **68 B/state**：64 packed e2m1 + 4 ue8m0 | fp32，**不含** q_scale | fp32 dense | ✅（非 H200） |
| `mxfp4` | `True` | sm100 | 同上 | 68 B/state | `(T,32)` **bf16** | **bf16** sparse（仅候选块） | ✅（非 H200） |
| `bf16` / `nvfp4` | 任意 | 任意 | — | — | — | — | ❌ `ValueError`（`mla/indexer.py:62-66`） |

依据：`mla/indexer.py:56`、`:61`、`:90-95`；`config/attention.py:13`、`:163-167`；`attention.py:92-100`、`:442`、`:1307`、`:1366-1370`。

> ⚠️ **与官方参考实现的差异**：`dsv41_part/inference/model.py:759` 注释称 *"Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0"* —— 参考实现的 indexer K 是 **FP4 E2M1 + 每 32 元素一个 UE8M0**（68 B 行），vLLM 的 `fp8` 选项是 **e4m3 + 每 128 元素一个 fp32**（132 B 行），**两者不是同一个量化方案**。vLLM 的 `mxfp4` 才贴近参考实现，但它 H200 不可用。这意味着 H200 上的 indexer top-k 与训练语义存在量化方案差异。

### 6.4 ★ KV cache 的数据布局（page / block / slot / 各占几字节）

```
[kv 张量]  [num_blocks, N, 132] uint8      N = P // r  (r=1 → 64, r=2 → 32)
           stride(0) = S_page = round_up(N * 132, 576)      H200: r=1 → 8640, r=2 → 4608
  block b ├─ VALUE PLANE : [b*S_page,              b*S_page + N*128)      128 B/state
          ├─ SCALE PLANE : [b*S_page + N*128,      b*S_page + N*128 + N*4)  4 B/state
          └─ padding     : [b*S_page + N*132,      b*S_page + S_page)
  slot 计算 (indexer_k_store.py:192-208)：
    N = k_cache.shape[1]          # 每 block 的 state 数 == P // r；也是分块除数
    base = (slot // N) * S_page ;  state = slot % N
    val_ptr   = base + state * 128
    scale_ptr = base + N * 128 + state * 4
  ⚠️ 分块与行内偏移都取 k_cache.shape[1]（= N），**不是** Python 侧的 P=64；
     r=1 时二者相等（64），r=2 时是 32 vs 64 —— 混用会算错 scale 平面地址。
```

**为什么是两个平面**（两条独立证据）：V3.2 起的 CUDA 写入 kernel `indexer_k_quant_and_cache_kernel`（device 端，`csrc/cache_kernels.cu:624`；宿主启动器 `indexer_k_quant_and_cache` 在 `:1917`）与 V4.1 的 Triton 写入 kernel（`indexer_k_store.py:202-208`）都写成「值平面 + 尺度平面」。

> ⚠️ **`utils/deep_gemm.py:716-718` 的注释与之矛盾**：它把 FP8 布局描述为 `[nb, bs, 1, D+4]`、「每 (block,pos) 的最后 4 字节是 scale」，即**每 state 连续的 132 字节行**。按行主序读会**静默错位**；参考实现自检 S7 断言两种读法结果不同。

**★ dense-logits 与工作区的尺寸（做 H200 容量规划时必看；上一版漏了）**

`sparse_attn_indexer.py` 里 dense logits 和 top-k 的工作区是**一次性预留**的，它们多大由一个环境变量说了算：

| 项 | 值 / 代码 |
|---|---|
| **`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`** | 唯一能调大小的开关；`max_logits_elems = VLLM_SPARSE_INDEXER_MAX_LOGITS_MB × 1024 × 1024`（`sparse_attn_indexer.py:465`），分配为 uint8（`:467`） |
| 预留时机 | **profiling run 期间**（`:445-461`），连同 K-gather 工作区（`:446` `_gather_workspace_shapes`）与 `RADIX_TOPK_WORKSPACE_SIZE = 1 MiB`（`:452`）一次性 `get_simultaneous(...)` |
| 交付给 kernel | `current_workspace_manager()` 在首次使用时分配（`:571-576`） |
| 为什么要关心它 | H200 上 `indexer_sparse_logits` **默认是关的**，于是只能走 **dense logits**（宽度是 `max_model_len // r`），这块内存**不受候选筛选保护**，只能靠这个环境变量兜住 |

> 换句话说：在 H200 上算 indexer 的显存预算时，**dense logits 工作区是一块独立于 KV cache 的常量**，想改它只能动这个环境变量。

**page 几何（`P = 64`）**

| `r` | 格式 | 未对齐 page | `S_page`（align 576） | H200 可达 |
|---|---|---|---|---|
| 1 | fp8 | 8448 | **8640** | ✅ |
| 1 | mxfp4 | 4352 | 4608 | ❌ |
| 2 | fp8 | 4224 | **4608** | ✅（L2/8/14） |
| 2 | mxfp4 | 2176 | 2304 | ❌ |

**576 这个对齐是怎么来的**（`attention.py:1225-1229` 注释）：indexer 的 page 和主 KV 的 page 共用同一个物理 block，block stride 是这一组里各个 page 之和，所以它得同时满足主 record 的 TMA stride 和对齐要求。H200 上 `_use_v41_mxfp8_kv_record() = False` → 主 record 是 584 B/token 的 V4 record → **576**。

**slot 映射链路**：metadata builder 出 `slot_mapping` → C3 写 K → 打分出 local compressed id → C7/C6 出 top-k → `compute_global_topk_indices_and_lens`（`slot = block_table[req, local // bs] * bs + (local % bs)`）→ `combine_topk_swa_indices` → `flash_mla_sparse_fwd`。

### 6.5 索引缓冲区由谁写、由谁读

- **只有一张共享缓冲区**：`topk_indices_buffer = (max_num_batched_tokens, 512) int32`（`nvidia/model.py:704-708`），8 个 indexer 层共用它；**里面不存任何 weights**。
- **四个写入者，互不重叠**：C9 的短上下文快路径 / C7 的 dense prefill / C6 的 dense decode / sparse 路径（H200 到不了）。
- **预清 `-1`**：`topk_indices_buffer[:hidden_states.shape[0]] = -1`（`sparse_attn_indexer.py:565-566`）；V4.1 走 `skip_k_cache_insert=True`（`attention.py:1392`）跳过重复 K 填充，但 `skip_topk_buffer_clear` 仍是 `False`（`:1104`）。
- **一个特例**：主 MLA 判定这批走 dense MHA 且无 decode token 时，op **刻意不动缓冲区**直接返回（`:545-558`）。
- **decode 的 padding 与 `next_n` 布局**：`num_padded_tokens = batch_size * next_n`，top-k 视图是 **batch-major**（第 `b*next_n + n` 行）；`seq_lens` 是 `(B, next_n)`（native spec decode）或 `(B, 1)`。`needs_padded_path = requires_padding or (num_decode_tokens % decode_lens.shape[0] != 0)`（`:754-756`），走 `pack_seq_triton` / `unpack_seq_triton`（`:888-897`）——**缓冲区最终只有真实 decode 行**。
- ⚠️ 预清只把 `[:hidden_states.shape[0]]` 这段刷成 `-1`：如果 padding 行比真实 token 还多，多出来的那几行会留着上一步的旧值。现在靠 `seq_lens` 来界定有效长度，这属于一条**隐含依赖**。

### 6.6 K cache 到底归哪一层（index-source 的依赖链）

```python
# attention.py:343-348
self.kv_source_layer_id    = max(s for s in kv_source_layers    if s <= layer_id)
self.index_source_layer_id = max(s for s in index_source_layers if s <= layer_id)
```

| index-source 层 | kv source? | K cache 归属 | 解析方式 |
|---|---|---|---|
| 2 / 8 / 14 / 20 | **是** | 自己 | `DeepseekV4IndexerCache(prefix=f"{prefix}.indexer.k_cache")`（`attention.py:437-446`） |
| 24 / 28 / 32 / 36 | 否 | **L20** | `_replace_layer_index(prefix, 20)` → 查 `static_forward_context`（`:449-453`） |

只有 **4 个** index K cache 张量；L24/28/32/36 的 `_produce_k` 从不执行（`owns_k=False`，`:1506`），它们打分用的是 **L20 的 key**（图形见 §7 图 X1）。compressed-KV cache 同理（`:588-603`、`:1165-1172`）；要是在 `static_forward_context` 里查不到源，就直接抛 `NotImplementedError`——也就是说 **v4.1 的 kv-sharing group 内部不支持 PP 切分**。

---

## 7. 结构图

本节 **7 张图**全部由 `..\indexer\_work\make_diagrams.py` 按固定几何**程序化生成**，并由同一脚本做对齐审计（判据：每个 `▼` / `▲` 的**正上方同索引位置**必须是 `│┌┐└┘┬┴┼─` 之一）。

**图里只放结构**：方块、箭头、短标识符，字形全是 ASCII/Latin（`▼`/`▲` 按 1 列计）；中文一律放在代码围栏**之外**，每张图下面紧跟的 **注记** 就是它的完整解释。这么做一是读图时不必在英文长句里找结构，二是「字符下标 == 显示列」成立，脚本可以逐格判定对齐（mHC 先例把 CJK 放进图内、靠脚本按东亚宽度排版；本子系统选了更保守的写法，代价是图内文案短，收益是审计精确）。

本节图块与 `python ..\indexer\_work\make_diagrams.py --emit-section` 的输出**逐字节一致**（由 `..\indexer\_work\diff_sec7.py` 断言），因此图不可能是手写对齐的。同一份图在 `..\indexer\INDEXER_diagram.md` 里另有一份独立副本（含对齐审计章节）。

**图目录**

1. **总览：位置、依赖与图索引**
2. **indexer Q 构造（FP8 分支）**
3. **indexer K 构造（FP8 分支）**
4. **dense 打分与 top-k（prefill / decode 双路）**
5. **两层候选筛选：发布层 vs 消费层**
6. **跨层 index-source 依赖链**
7. **分页 K cache 字节布局（page / block / slot）**

### 总览：indexer 的位置、依赖与图索引

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ index_source_layer_ids = {2, 8, 14, 20, 24, 28, 32, 36}   (8 layers)               │
  │ kv_source_layer_ids    = {2, 8, 14, 20}   (only these own a K cache)               │
  │ compress_ratio: L2/8/14 = 2,  L20/24/28/32/36 = 1,  L0/1 + mtp = 0                 │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ hidden_states (T, 5120) --[C1] weights_proj--> indexer_weights (T, 32)             │
  │ qr (T, 1280)            --[C1] wq_b (MXFP8)--> index_q (T, 32, 128)                │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ [C2] fused_indexer_q_rope_quant   (FP8 branch = the one H200 takes)                │
  │        -> q_fp8 (T, 32, 128) float8_e4m3fn   (no companion scale)                  │
  │        -> weights_out (T, 32) float32        (q_scale folded in)                   │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ [C3] indexer_k_norm_rope_store    (FP8 branch = the one H200 takes)                │
  │        latent (T, 512) -> wk -> k_norm -> RoPE@group-start -> e4m3                 │
  │        -> paged index K cache [num_blocks, P//r, 132] uint8                        │
  │        written only by kv-source layers, only at group boundaries                  │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ [C5] dense logits = sum_h w_h * relu(q_h . k)   fp32, causal-masked                │
  │ [C8] candidate publish (L20) / mask (L24/28/32/36)  int32 block ids                │
  │ [C6] decode top-k: cooperative -> persistent -> per_row  (H200: coop)              │
  │ [C7] prefill top-k: ops.top_k_per_row_prefill                                      │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ topk_indices_buffer (max_num_batched_tokens, 512) int32                            │
  │ LOCAL compressed-position ids, -1 padding, no weights persisted                    │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ [C10] compute_global_topk_indices_and_lens -> global KV slots + lens               │
  │       combine_topk_swa_indices -> merged with the SWA window                       │
  │       slot = block_table[req, local // bs] * bs + (local % bs)                     │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ [C12] FLASHMLA_SPARSE_DSV41  sparse MLA attention (the only H200 one)              │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure X0: the topology that decides which of the 12
# structure classes exist on which layer. No compute at all.
index_source = (2, 8, 14, 20, 24, 28, 32, 36)   # has an indexer -> C1..C3, C5..C10
kv_source    = (2, 8, 14, 20)                   # owns the K cache -> C3, C4
ratios       = (0, 0, *([2] * 18), *([1] * 20), 0, 0, 0)          # len 43
candidate_source_layer_id, candidate_topk_blocks, candidate_block_size = 20, 2048, 8
```

#### 注记

- 12 个结构类在这一张图里各出现一次；后面 6 张图按 `C1 → C2 → C3/C5/C6/C7 → C8 → C10/C12` 依次展开，跨层的耦合只在图 X1 里讲。
- 图与结构类怎么对上：`C1`/`C2` → 图 C1；`C3` → 图 C2；`C5`/`C6`/`C7` → 图 C3；`C8` → 图 C4；`C4` 的分页几何 → 图 X2；`C1`/`C3` 的跨层归属与 `C8` 的发布/消费角色 → 图 X1。`C9`（短上下文快路径）与 `C11`（后端 / metadata / 分片）**没有单独画图**：`C9` 的语义就是「什么都不做、直接返回」，`C11` 是配置与元数据，两者看 §4 对应的那一小节。
- H200 上走不到的路径不画进图，只在 §2.3 那张表里列出：mxfp4 K cache（68 B 行）、`indexer_sparse_logits` / `SparseMQAIndexer` / DeepSelect、ROCm 的 tiled(`SHUFFLE`) 与 aiter、XPU/CPU 的 forward、`FLASHINFER_MLA_SPARSE_DSV41`、`FLASHMLA_MEGA_ATTN_DSV41`，以及 DCP 的 merge / gather 辅助函数。

### 图 C1 — indexer Q 构造（`fused_indexer_q_rope_quant`，FP8 分支）

```text
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ IN  qr  (T, 1280) bf16   =  q_norm(wq_a(hidden_states))                      │
    └──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ wq_b  ReplicatedLinear(1280 -> 32*128 = 4096)   MXFP8 weight+act             │
    └──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ view(-1, 32, 128)  ->  index_q (T, 32, 128) bf16                             │
    │ positions (T,) int64 ; cos_sin_cache (max_pos, 64) fp32 GPT-J                │
    └──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ fused_indexer_q_rope_quant(use_fp4=False, weights_out_dtype=float32)         │
    │                                                                              │
    │   dims [0,64)   : NoPE, passed through unchanged                             │
    │   dims [64,128) : GPT-J interleaved RoPE at `position`,                      │
    │                   fp32 -> bf16 -> fp32 round trip on the pairs               │
    │   amax          = max(|rope|, |nope|)                                        │
    │   q_scale       = 2 ** ceil(log2(max(amax, 1e-4) / 448))   <- power of two   │
    │   q_fp8         = (x / q_scale) -> float8_e4m3fn                             │
    │   w_out         = w * q_scale * softmax_scale * head_scale                   │
    │                   softmax_scale = 128**-0.5 = 0.088388                       │
    │                   head_scale    =  32**-0.5 = 0.176777                       │
    └──────────────────────────────────────────────────────────────────────────────┘
          ┬───────────────────────────────────────┬
          │                                       │
          ▼                                       ▼
    ┌──────────────────────────────────────┐  ┌──────────────────────────────────────────┐
    │ q_quant = q_fp8                      │  │ weights_out (T, 32) fp32                 │
    │ (T, 32, 128) e4m3fn                  │  │ q_scale is FOLDED IN                     │
    │ NO companion scale                   │  │ -> fp8_fp4_mqa_logits /                  │
    │ tensor is returned                   │  │    _paged_mqa_logits eat fp32            │
    └──────────────────────────────────────┘  └──────────────────────────────────────────┘
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure C1 (the FP8 branch).
qs    = index_q.float()                                    # wq_b(qr).view(T, H, D)
roped = gptj_rope_forward(qs, pos, cos_sin, rope_dim=64)   # dims [64,128) only
roped = roped.to(torch.bfloat16).to(torch.float32)         # kernel round trip
q_scale = 2 ** torch.ceil(torch.log2(
    torch.clamp(roped.abs().amax(-1), min=1e-4) / 448.0))   # power of two
q_fp8 = e4m3_quantize(roped / q_scale[..., None])           # (T, H, D), e4m3fn
weights_out = w.float() * q_scale * softmax_scale * head_scale   # q_scale FOLDED IN
# floor is 1e-4 (fused_indexer_q.py:163), NOT the 1e-10 of the V3.2 sibling at
# sparse_attn_indexer.py:211; and q_scale is folded into weights_out.
```

#### 注记

- H200 上到底走哪个 kernel 由环境决定：`has_cutedsl()` 为真就走 CuTeDSL 的 `IndexerQFp8Kernel`，否则走 `FusedIndexerQRopeQuantTritonKernel`，两者实现的是同一套算术。
- `q_scale` 只活在 `fused_indexer_q_rope_quant` 内部，而且**被折进 `weights_out`**，不是一张独立张量；它和 `wq_b` 的 MXFP8 **激活**量化（在 Linear 内部做）是两回事。
- `use_fp4=True` 那条分支在 sm90 上到不了：它返回的是 `(q_packed, q_scale)` 二元组、**不**把 `q_scale` 折进 weights，但 `dsa_indexer_uses_fp4()` 在 sm90 上遇到 mxfp4 会先抛 `ValueError`。

### 图 C2 — indexer K 构造（`indexer_k_norm_rope_store`，FP8 分支）

```text
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ IN  latent (T, 512) bf16  =  kv-source compressor's pre-RoPE latent                │
    └────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ wk  ReplicatedLinear(512 -> 128)   bf16, quant_config=None                         │
    └────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ k_pre (T, 128) bf16   -- only group-boundary rows are read                         │
    └────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ indexer_k_norm_rope_store(k_pre, positions, cos_sin, k_norm.w, eps,                │
    │                            k_cache, slot_mapping, compress_ratio, use_fp4)         │
    │                                                                                    │
    │   one program per token; early-exit when  slot < 0                                 │
    │   early-exit when  (position + 1) % compress_ratio != 0     <- GROUP GATE          │
    │   k = RMSNorm(k_pre), with a fp32 -> bf16 -> fp32 round trip                       │
    │   RoPE at compressed_pos = (position // ratio) * ratio      <- GROUP START         │
    │   fp32 -> bf16 -> fp32 round trip on the rotated pairs                             │
    │   FP8 (the H200 branch): absmax over the whole 128-wide row, floor 1e-4            │
    │       scale = 2 ** ceil(log2(absmax / 448))   (ONE fp32 per state)                 │
    │       x_uint8 = clamp(x / scale, -448, 448) -> e4m3fn                              │
    └────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ paged indexer K cache  torch.uint8  [num_blocks, block_size // ratio, 132]         │
    │   value plane @ blk*stride + state*128         : 128 e4m3 bytes                    │
    │   scale plane @ blk*stride + N*128 + state*4   : 1 fp32 (4 bytes)                  │
    │   block stride = round_up(N*132, 576) = 8640 for N = 64 on H200                    │
    └────────────────────────────────────────────────────────────────────────────────────┘
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure C2 (the FP8 branch).
k = k_pre.float()
k = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps) * k_norm_w)
k = k.to(torch.bfloat16).to(torch.float32)                  # RMSNorm round trip
group_pos = (positions // compress_ratio) * compress_ratio  # GROUP START
k = gptj_rope_forward(k, group_pos, cos_sin, rope_dim=64).to(torch.bfloat16).float()
emit  = (positions + 1) % compress_ratio == 0               # GROUP GATE
scale = 2 ** torch.ceil(torch.log2(torch.clamp(k.abs().amax(-1), min=1e-4) / 448.0))
codes = e4m3_code(torch.clamp(k / scale[..., None], -448.0, 448.0))
buf[blk*stride + s*128 : +128] = codes[s]                   # value plane
buf[blk*stride + N*128 + s*4 : +4].view(torch.float32)[0] = scale[s]   # scale plane
```

#### 注记

- `wk(latent)` 这个 GEMM 走 torch；`k_norm → RoPE → 量化 → 分页写入` 四步由一个 Triton kernel 融合，**原地写入、没有返回值**，而且必须在 `indexer_op` 读它之前写完。
- **组语义**：只有 `(pos+1) % ratio == 0` 的行才发布，RoPE 取的是**组起点**；每 128 宽的一行只配一个 2 的幂 fp32 scale。
- `use_fp4=True` 那个兄弟分支写的是 64 个打包的 e2m1 字节 + 4 个 ue8m0 字节，在 H200 上同样到不了（`dsa_indexer_uses_fp4()` 会先抛 `ValueError`）。

### 图 C3 — dense 打分与 top-k（prefill / decode 双路）

```text
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ sparse_attn_indexer custom op   (mutates topk_indices_buffer)                      │
    └────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │ topk_indices_buffer[: num_tokens] = -1    <- pre-clear; -1 is the sentinel         │
    │ asserts: use_fp4_cache is False ; q_scale is None        (FP8 contract)            │
    │ when skip_k_cache_insert is False: gather K, quantize, insert                      │
    └────────────────────────────────────────────────────────────────────────────────────┘
          ┬───────────┬───────────────────────────────────────────┬
                      │                                           │
                      ▼                                           ▼
  ┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
  │ PREFILL (per chunk)                    │  │ DECODE (per padded row)                │
  │  cp_gather_indexer_k_quant_cache       │  │  kv_cache_as_quant_view ->             │
  │    -> k_quant (T,128) fp8              │  │    [nb, bs, 1, 132] uint8              │
  │       k_scale (T,4) fp32-in-u8         │  │  fp8_fp4_paged_mqa_logits(             │
  │  fp8_fp4_mqa_logits(                   │  │    (q, None), kv, w, seq_lens,         │
  │    (q_fp8, None), (k, ks), w,          │  │    block_table, schedule_metadata,     │
  │    cu_seqlen_ks, cu_seqlen_ke)         │  │    max_model_len)                      │
  │  logits (rows, width) fp32,            │  │  logits [num_padded_tokens,            │
  │    -inf outside [ks, ke)               │  │    max_model_len // ratio] fp32        │
  └────────────────────────────────────────┘  └────────────────────────────────────────┘
  ┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
  │ ops.top_k_per_row_prefill              │  │ get_indexer_topk(backend)              │
  │    (logits, ks, ke, out, ...)          │  │   auto: cooperative (rows<=64)         │
  │  -> int32, -1 padded                   │  │      -> persistent -> per_row          │
  └────────────────────────────────────────┘  └────────────────────────────────────────┘
                      │                                           │
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ OUT  topk_indices_buffer (max_num_batched_tokens, 512) int32                       │
  │      LOCAL compressed-position ids, -1 padding; NO weights persisted               │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure C3.
q      = e4m3_dequantize(q_fp8)                        # q_scale is None (folded in)
qk     = torch.einsum("thd,sd->ths", q, k_deq)         # (T, H, S)
logits = (torch.relu(qk) * weights[..., None]).sum(1)  # ReLU is PER HEAD
cols   = torch.arange(k_deq.shape[0])
logits = logits.masked_fill(~((cols >= ks[:, None]) & (cols < ke[:, None])), -inf)
topk   = top_k_per_row(logits, 512)                    # int32, -1 padded
```

#### 注记

- `ReLU` 是在 DeepGEMM kernel **内部、按 head** 做的：`Σ_h w_h·relu(q_h·k)`。如果写成「先把各 head 求和、最后再 ReLU」，选出来的 top-k 就不一样了。
- prefill 走 `fp8_fp4_mqa_logits`（先把 K gather 成连续内存）；decode 走 `fp8_fp4_paged_mqa_logits`（直接读分页 cache）。H200 上永远是 dense logits。
- 两条路最后都写进 `topk_indices_buffer`：prefill 用 `ops.top_k_per_row_prefill`；decode 用 `get_indexer_topk`，`auto` 的顺序是 cooperative → persistent → per_row；cooperative 要求行数 ≤ 64，行数少时第一档就命中，否则往后落。
- 写进去的是 **local** compressed id，后面由 `compute_global_topk_indices_and_lens` 换算成全局 KV slot：`slot = block_table[req, local // bs] * bs + (local % bs)`，原来是 `-1` 的位置仍然是 `-1`。

### 图 C4 — 两层候选筛选：发布层 vs 消费层

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ logits (rows, width) fp32, already -inf outside [ks, ke)                           │
  │ width = nblocks * BLOCK                                                            │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  publisher: layer 20 (candidate_write=True)
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ _select_candidate_blocks   candidate_blocks.py:141                                 │
  │   _block_scores_kernel : NaN-propagating max over each block                       │
  │     the row's NEWEST block ((end - start - 1) // BLOCK) -> +inf                    │
  │   scores.topk(min(2048, nblocks))                                                  │
  │   _store_candidates_kernel : value != -inf -> block id, else -1                    │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ candidate_block_buffer  (max_num_batched_tokens, 2048) int32, -1 padded            │
  │ ONE tensor shared by the whole model; the op declares mutates_args,                │
  │ so it is written in place                                                          │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  consumers: layers 24, 28, 32, 36 (candidate_write=False)
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ _apply_candidate_mask   candidate_blocks.py:192                                    │
  │   _candidate_flags_kernel : flags uint8 [rows, nblocks+1]                          │
  │     a candidate whose first column >= width clamps to slot nblocks                 │
  │   _mask_candidates_kernel :                                                        │
  │     keep  = flags[block] ; keep |= (col == width-1) & flags[nblocks]               │
  │     logits = -inf unless (causal & keep)                                           │
  └────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ this layer's own top-k runs AFTER the mask  ->  topk_indices_buffer                │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of the candidate publish / consume pair.
nblocks = math.ceil(width / BLOCK)
scores  = blockwise_max(logits, BLOCK)                 # NaN-propagating block max
scores[..., (end - start - 1) // BLOCK] = float("inf") # PIN the newest block
vals, idx = torch.topk(scores, min(2048, nblocks), dim=-1)
cands = torch.where(vals != float("-inf"), idx, -1)    # -1 padding
flags = torch.zeros(rows, nblocks + 1, dtype=torch.uint8)          # nblocks+1 slots
for b in cands.tolist():
    if b >= 0:
        flags[row, nblocks if start + b * BLOCK >= width else b] = 1   # clamp + edge
keep = flags[row, block_of(col)].bool() | ((col == width - 1) & flags[row, nblocks].bool())
logits = torch.where(causal & keep, logits, -inf)      # then the consumer's top-k
```

#### 注记

- 缓冲区是**全模型共享的同一张张量**，op 用 `mutates_args` **原地改写**；发布层和消费层在同一个 forward step 内按层序读写它。
- **为什么要开 `nblocks+1` 个槽位**：越界的候选必须落到一个不是真实 block 的 id 上，mask kernel 再把这个哨兵读回来当 `edge` 标志，用来决定要不要钉住本请求的最后一列。但这条规则救不了「最后一列本身就不因果」的情况，因为写入侧的 mask 是 `~(valid & keep)`。
- **一条没被代码强制的不变量**：候选 id 用的是发布层的 compressed-position 坐标系，而代码里**没有任何 assert** 把消费者的 `compress_ratio` 和发布层绑在一起——它之所以正确，只是因为出厂配置里 L20 与 24/28/32/36 恰好都是 ratio 1。另外，DCP 下开候选筛选会被 assert 直接拒掉。

### 图 X1 — 跨层 index-source 依赖链（候选发布层 vs 消费层）

```text
                    L2       L8       L14      L20      L24      L28      L32      L36

ratio               2        2        2        1        1        1        1        1
is_kv_source        yes      yes      yes      yes      no       no       no       no
is_index_source     yes      yes      yes      yes      yes      yes      yes      yes
owns_k              yes      yes      yes      yes      no       no       no       no
K cache owner       L2       L8       L14      L20      L20      L20      L20      L20
candidate role      -        -        -        pub      con      con      con      con
compressor          yes      yes      yes      yes      no       no       no       no
                                               │
consumes K from                                ▲────────┬────────┬────────┬────────┬
                                                        │        │        │        │
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure X1: ownership resolution, no compute.
index_source = (2, 8, 14, 20, 24, 28, 32, 36)
kv_source    = (2, 8, 14, 20)
owner = {L: max(s for s in kv_source if s <= L) for L in index_source}
# owner == {2: 2, 8: 8, 14: 14, 20: 20, 24: 20, 28: 20, 32: 20, 36: 20}
owns_k          = {L: L in kv_source for L in index_source}
candidate_write = {L: L == 20 for L in index_source}
# the four consumers share owner 20 -> share its K cache AND its compressed-position grid
```

#### 注记

- L2/8/14/20 各建一份自己的 `{prefix}.indexer.k_cache` 并负责写入；L24/28/32/36 则靠改写 prefix 里的层号（`_replace_layer_index`）去 `static_forward_context` 里查 **L20 的 cache**，查不到就抛 `NotImplementedError`——也就是说 v4.1 的 kv-sharing group 内部不支持 PP 切分。
- L24..L36 的 `owns_k = False`，所以它们**既没有 `wk` 也没有 `k_norm`**，从不写 K；它们的全部工作就是拿 L20 的 key 打分、刷新自己的 top-k。
- L20 还额外**发布**候选块，L24..L36 先拿它做 mask；L2/8/14 在发布层之前，拿到的 `candidate_block_buffer` 是 `None`，走纯 dense。
- `topk_indices_buffer`（`(max_num_batched_tokens, 512)` int32）是 8 个 indexer **共用的一张**，每个 indexer 只覆盖自己那几行 token。
- **明确不涉及的层**：L0/L1（ratio 0）和 3 个 MTP 层（mtp.0/1/2，ratio 0）既没有 indexer 也没有 compressor；检查点里 `mtp.*` 下的 indexer key 是 0 个。

### 图 X2 — 分页 K cache 字节布局（page / block / slot）

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ One physical block of the indexer K cache, H200 (sm90, family(100) = False)        │
  │ block_size = 64 tokens, compress_ratio = 1  =>  N = 64 stored states               │
  └────────────────────────────────────────────────────────────────────────────────────┘

      byte 0              8192                8448                8640
      ├───────────────────┼───────────────────┼───────────────────┤
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼

  [0, 8192)     VALUE PLANE   64 states x 128 e4m3 bytes = 8192 B
  [8192, 8448)  SCALE PLANE   64 states x   4 bytes fp32 =  256 B
  [8448, 8640)  PADDING to the 576 B page alignment       =  192 B

  state s values : base + s * 128          (token_stride = 128)
  state s scale  : base + N * 128 + s * 4  (scale plane, NOT interleaved)
  base           : (slot // N) * stride(0) ;  N = k_cache.shape[1]
  stride(0)      : 8640 = round_up(64 * 132, 576)   (132 = 128 + 4)


Page geometry (block_size = 64):
  ratio   format   unpadded bytes   stride (align 576)   on H200
    1     fp8           8448              8640            YES
    1     mxfp4         4352              4608            no  (family(100))
    2     fp8           4224              4608            YES  (L2/8/14)
    2     mxfp4         2176              2304            no
```

#### 等价的纯 Torch 写法

```python
# Pure-torch equivalent of figure X2.
N         = block_size // compress_ratio      # == k_cache.shape[1]; 64 at r=1, 32 at r=2
row_bytes = 132 if not use_fp4_kv else 68     # AGGREGATE per state (not a row address)
stride    = round_up(N * row_bytes, 576)      # 576 on H200, 512 on sm100-class
# the kernel divides the slot by k_cache.shape[1] (== N), NOT by block_size;
# the two coincide only at r = 1.
base      = (slot // N) * stride
val_off   = base + (slot % N) * 128            # e4m3 value plane
scale_off = base + N * 128 + (slot % N) * 4    # fp32 scale plane
buf[val_off : val_off + 128] = codes
buf[scale_off : scale_off + 4].view(torch.float32)[0] = scale
```

#### 注记

- `132 = 128 + 4` 是**聚合**宽度，不是 scale 的地址：真实布局是**值平面 + 尺度平面**（尺度从 `N*128` 起），block stride 被 padding 到 576。
- 分块和行内偏移用的都是 `k_cache.shape[1]`（= N = `block_size // r`），**不是** Python 侧的 `block_size`——两者只在 r=1 时相等（r=2 时是 32 vs 64）。
- **两个写入实现写的是同一个布局**：`csrc/cache_kernels.cu:672-683`（`dst_offset = blk*stride + off*head_dim + d`，`dst_scale = blk*stride + bs*head_dim + (off*head_dim + d)*4/qbs`）与 `indexer_k_store.py:202-208`（`val_ptr = base + pos*TOKEN_STRIDE`，`scale_ptr = base + kv_cache_block_size*TOKEN_STRIDE + pos*SCALE_DIM`）。
- **读的一方也得按这个布局来**：`ops.cp_gather_indexer_k_quant_cache`（prefill 前 gather 成 `(T,128)` fp8 + `(T,4)` u8）和 DeepGEMM 的 `fp8_fp4_paged_mqa_logits`（decode，直接用 `[nb, bs, 1, 132]` 视图）。
- ⚠️ `utils/deep_gemm.py:716-718` 的注释把 FP8 布局写成「每 (block,pos) 的最后 4 字节是 scale」，也就是每 token 一行、值和尺度交错——这和上面两个写入实现**互相矛盾**，照这个注释按行主序去读会**静默读错 scale**。参考实现自检 S7 就是用来证明两种读法结果不同的。

---

## 8. 和上一代 DeepSeek-V4 的差异，以及同仓库里同名 indexer 的鉴别

### 8.1 结构性 diff（V4 = `vllm/models/deepseek_v4/`；V4.1 = `vllm/models/deepseek_v41/`）

| 方面 | V4 | **V4.1** | 依据（V4 → V4.1） |
|---|---|---|---|
| **K 的产生** | indexer **自带** compressor，做一次真实的 **`hidden_states` GEMM**（`fused_wkv_wgate` 权重 `[2·coff·head_dim, hidden_size]` = **`[256, 5120]`**；`256` 是输出维，第二维是模型 hidden size） | indexer **无 compressor**；K = `k_norm(wk(latent))`，`latent` 是 kv-source compressor 的 pre-RoPE 512 维输出；**无 hidden_states 上的 K GEMM** | `deepseek_v4/compressor.py:271-273`、`deepseek_v4/attention.py:1022-1031`、`:711-719` → `deepseek_v41/attention.py:943-947`（注释）、`:1405-1437`、`indexer_k_store.py:1-14` |
| **indexer 权重量化** | **只有 `weights_proj` 是 `quant_config=None`；`wq_b` 是量化的**（`quant_config=quant_config`）；**无** `wk`/`k_norm` | 两者同构：`wq_b` 量化（MXFP8）、`weights_proj` 不量化；**新增** `wk` 与 `k_norm`（仅 `owns_k`，**均不量化**，检查点里 `wk` 是 bf16 无 scale） | `deepseek_v4/attention.py:975-981`（量化）vs `:982-988`（None）→ `deepseek_v41/attention.py:1314-1320` vs `:1321-1327`、`:1344-1354` |
| **`compress_ratio` 语义** | `max(1, cr[i])`，值域 `{4,128}`；**SWA-only = `<= 1`**；`get_kv_cache_spec` 在 `<= 1` 返回 `None`；有 C128A 分支 | 保存原始 int（0 合法，`not in (0,1,2)` 抛异常）；**SWA-only = `== 0`**；**无 128** | `deepseek_v4/nvidia/flashmla.py:109` → `deepseek_v41/nvidia/flashmla.py:98`；`deepseek_v41/attention.py:310-321` |
| **算子集** | 用 `compress_norm_rope_store_triton`、`_SAVE_PARTIAL_STATES_KERNEL`、第 3 个 hidden-state GEMM；**不用** `indexer_k_norm_rope_store` / `SparseMQAIndexer` / 候选 | **新**：`indexer_k_norm_rope_store`、`wk` GEMM、`SparseMQAIndexer`、候选 select/apply；**从 V4 树复用**：`fused_indexer_q`、`fused_inv_rope_fp8_quant` | `deepseek_v4/compressor.py:20-22`、`:472` → `deepseek_v41/common/ops/__init__.py:5-30` |
| **cache 布局** | 压缩 record 只有 584 B；indexer K cache 132/68 B（公式**内联**）；backend block size **256** | 584 / 528 / 288 B 三常量；`swa_bytes_per_token=584`、`kv_page_alignment=576`（H200）；**新增** `block_stride_alignment`；block size **64(sm90)/128** | `deepseek_v4/attention.py:1005-1014` → `deepseek_v41/attention.py:92-100`、`:512-527`、`:1216-1238`；`mla/indexer.py:366-372` |
| **两层候选筛选** | **不存在**（V4 树中 `candidate_*` 符号 0 命中） | **V4.1 独有**（§7 图 C4） | `deepseek_v41/attention.py:460-482`；`candidate_blocks.py:141`、`:192` |
| **跨层拓扑** | **无**：每层各建自己的 indexer / K cache / compressor | 显式 producer/consumer；`kv_source_layer_ids` / `index_source_layer_ids` / `owns_k` / `candidate_source_layer_id` 全部存在 | `deepseek_v41/attention.py:303-351`、`:432-482` |
| **检查点兼容性** | indexer 子树含 `attn.indexer.compressor.fused_wkv_wgate.*` | 通用加载 `indexer.wk.weight` / `indexer.k_norm.weight`；**两代 indexer 权重不兼容** | `deepseek_v4/nvidia/model.py:1530-1538` → `deepseek_v41/nvidia/model.py:1251`（`params_dict = dict(self.named_parameters())`；注意 `:1242-1250` 的 `stacked_params_mapping` **不含** indexer 的 wk/k_norm 条目） |

⚠️ V4.1 里仍有一处 `<= 1`（`nvidia/model.py:811`），但它是保护 SWA bounded replay 的 **assert，不是 gate**。
⚠️ `deepseek_v4/attention.py:948` 的注释写 `index_n_heads  # 64`，但 V4.1 是 **32**（`deepseek_v41/attention.py:1301` 与检查点均为 32）——陈旧注释，跨代读代码时易被误导。

### 8.2 同仓库其他模型族的 indexer（★ 必须分清，别混）

怎么区分：看**打分公式、聚合方式、top-k 的单元，以及是否共用代码对象**这四点。

| 模型族 | 入口 | 打分 | 聚合 | top-k 单元 | 与 DSV4.1 共用代码？ | 判定 | H200 |
|---|---|---|---|---|---|---|---|
| **deepseek_v41** | `attention.py:1255` | `Σ_h w_h·relu(q_h·k)` | 按 head 加权 | token（压缩位置） | — | **参考对象** | ✅ |
| deepseek_v32 / v4 / hy_v4 | `deepseek_v32/attention.py:50`；`deepseek_v4/attention.py:928`；`hy_v4/nvidia/attention.py:133` | 同上 | 同上 | token | **是**——同一个 `SparseAttnIndexer` 对象（`deepseek_v4/attention.py:1033`、`model_executor/models/deepseek_v2.py:742`）+ 同一个 `indexer_topk.py` | **同一算法** | ✅ |
| **glm5next**（kpool） | `glm5next/common/attention.py:187`；`nvidia/sparse_indexer.py:677`（类 `SparseAttnIndexerKpool`） | **同一** `Σ_h w_h·relu(q_h·kpool[p])`，走**同一对** DeepGEMM kernel（import 在 `:347`、`:597`；调用在 `:348`、`:598`） | 按 head 加权 | **pool**（`select_k = topk // index_kpool`，`:613`） | **部分**：同用 `ops.indexer_k_quant_and_cache`（`:251`）、`get_indexer_topk`、`PrefillRowShard`、`_merge_dcp_topk_global`、`DeepseekV32IndexerMetadata`；且 `Glm5NextIndexerBackend` **就定义在 DeepSeek 的文件里**（`mla/indexer.py:296-306`） | **同打分算法，不同 key 空间 + pool 粒度选择** | ✅ |
| **minimax_m3** | `minimax_m3/common/indexer.py:553` | `score[h,t,blk] = max_{s∈blk}(q_h·k)`——**无 relu、无 weights、不跨 head 求和**（`common/ops/index_topk.py:171-176`） | **无**（按 index head 各做各的 top-k） | 128-token block（`SPARSE_BLOCK_SIZE=128`） | **否**（`vllm/models/minimax_m3/` 下 grep `sparse_attn_indexer\|indexer_topk` = 0）；自有 backend / cache spec / builder / kernel | **不同算法** | ✅（Triton 实现无架构门） |
| **qwen4_exp**（QSA） | `qwen4_exp/nvidia/indexer_qsa.py:90` | `Σ_h relu(q_h·k)`——**weight-free**，无 `weights_proj`（`nvidia/ops/qsa_indexer.py:94-100`） | 不加权 relu 求和 | 压缩块（`block_topk = token_topk // compress_ratio`） | **否**——直接调 `torch.ops._C.cooperative_topk` / `persistent_topk`，**绕过** `SparseIndexerTopk`（`:486-505`） | **不同算法**（最接近的亲戚） | ✅ |
| kimi_k3 / inkling | — | — | — | — | — | **根本没有 indexer**（`kimi_k3/nvidia/mla.py:355` 显式 `indexer=None`） | — |

**`SparseIndexerTopk` 的用户**：只有 DeepSeek DSA 栈（`sparse_attn_indexer.py:32-35`、`:869-875`）与 **glm5next 两个平台**（`glm5next/nvidia/sparse_indexer.py:14`、`:626`）；minimax_m3 与 qwen4_exp 完全不用。

---

## 9. 单元测试覆盖到什么程度

### 9.1 测试资产总览

**13 个文件名含 `indexer` 的文件**（实测合计 **5010 行 / 86 个 `def test_*`**）

| 文件 | 行数 | 测试 | H200 状态 |
|---|---|---|---|
| `tests/kernels/test_fused_indexer_q_rope_quant.py` | 615 | 6 | fp8 参数运行；MXFP4 参数 `:140-146` skip；**`:566-572` 是单测试 `test_fused_indexer_q_rope_quant_writes_bf16_weights` 的 `skipif`**（SM100 门控，H200 跳过整个测试，**不是整文件 skip**）；ROCm 用例 `:497` skip |
| `tests/model_executor/layers/test_mla_short_prefill_indexer.py` | 1061 | 13 | 10 运行；3 个 SM100 用例 skip（`:722-734`）；`:689` 需 `_deepselect_C` |
| `tests/v1/attention/test_indexer_dcp_localize.py` | 1036 | 18 | 15 运行；`:517`、`:436`、`:911` 需 CuTeDSL |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | 656 | 15 | 14 运行；`:105` 无 DeepGEMM 时 skip |
| `tests/v1/attention/test_indexer_tp_row_shard.py` | 213 | 5 | 全部运行（**mock/CPU**） |
| `tests/v1/attention/test_indexer_native_next_n.py` | 75 | 4 | 全部运行（`:63` 是显式 sm90 用例） |
| `tests/v1/attention/test_kpool_indexer_block_sizes.py` | 121 | 2 | 运行（CPU）—— **GLM 族** |
| `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py` | 188 | 7 | 运行（CPU）—— **GLM 族** |
| `tests/v1/attention/test_glm5next_indexer_block_sizes.py` | 40 | 1 | 运行（CPU）—— **GLM 族** |
| `tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py` | 599 | 5 | **整个模块 skip**（`:20` ROCm + `:43` CDNA4） |
| `tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py` | 162 | 4 | **整个模块 skip**（`:15` ROCm） |
| `tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py` | 143 | 3 | 全部运行—— **MiniMax M3，不同算法** |
| `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` | 101 | 3 | 1 运行（需 DeepGEMM）；`:46`/`:58` ROCm skip |

**3 个「文件名不含 indexer 但测了 indexer」的相邻文件**（K 侧规格与 top-k 离不开它们）

| 文件 | 行数 | 测试 | 说明 |
|---|---|---|---|
| `tests/kernels/test_compressor_kv_cache.py` | 2423 | 25（6 个 indexer） | `:1425` 量化往返、`:1489` gather、`:1752` fused KV insert（`torch.equal` 做 bitwise 断言 `:1874`/`:1882`）、`:2370` `test_v41_indexer_forward_q` |
| `tests/kernels/test_deepseek_v4_cpu_kernels.py` | 1494 | 16（3 个 indexer） | CPU 版 indexer 路径；**H200 上整体 skip**（`:55-68`） |
| `tests/kernels/test_top_k_per_row.py` | 2036 | 34（10 个 indexer） | `:815` dsv4 window、`:1707+` decode top-k、`:1816` sm90 分支 |

### 9.2 逐个看子部件的覆盖（H200 视角）

| 子部件 | 覆盖 | 关键用例 |
|---|---|---|
| (a) Q 侧 rope + quant | 强 | `test_fused_indexer_q_rope_quant.py:153`（fp8 全跑） |
| (b) K 侧 norm+rope+quant+store | 强 | 同上 `:359`、`:444`；`test_compressor_kv_cache.py:1752` |
| (c) K cache 布局 / slot mapping | 强 | `test_compressor_kv_cache.py:1425`、`:1489` |
| (d) top-k / block size | 强 | `test_top_k_per_row.py:815`、`:1707+`、`:1816` |
| (e) **两层候选筛选** | **部分** | 子 kernel 在 sm90 上跑（`:406`、`:588`、`:640`），但**端到端只有 ROCm 用例**（`tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py:429`、`:463`；`:320-418` 是 `_run_layers` 辅助函数）；消费侧端到端被 SM100 门控 |
| (f) DCP / TP 分片 | DCP 好；TP **仅 mock** | `test_indexer_dcp_localize.py`；`test_indexer_tp_row_shard.py:67-74` |
| (g) decode vs prefill | 好 | `test_indexer_native_next_n.py` |
| (h) MXFP4 / ROCm 变体 | 充分但 **H200 全 skip** | `test_rocm_paged_mxfp4_indexer*.py` |
| (i) **cache 格式选择** | **零覆盖** | 见 §9.3 第 1 条 |

### 9.3 没覆盖到的地方

| # | 缺口 | 证据 |
|---|---|---|
| 1 | **`dsa_indexer_uses_fp4()` 无任何直接测试**，5 个 `ValueError` 分支全未覆盖（含 sm90 拒绝分支） | `tests/` 下 grep 该函数 = 0；抛出点在 `mla/indexer.py:62-66`、`:75-79`、`:80-83`、`:84-88`、`:90-95`；唯一的 mxfp4 配置用例都在 CDNA4 门控下 |
| 2 | **V4.1 的 K 写入方 `indexer_k_norm_rope_store` 在 CUDA 上有写入侧逐字节验证，但缺「经消费者路径读回」的验证** | ⚠️ **此前本条写作"在 CUDA 上从未被读回验证"，那是错的**（且与 §9.2(b) 自相矛盾）：`test_fused_indexer_q_rope_quant.py:359`（`device="cuda"` `:370`，调用 `:414-424`）与 `:444` 都真实调用 V4.1 写入方并做 **`assert_close(storage, expected, rtol=0, atol=0)`**（`:438`）。**真正的缺口是**：没有 CUDA 测试把写好的 K 行经**消费端**（`cp_gather_indexer_k_quant_cache` + `fp8_fp4_mqa_logits`）读回并比对——唯一的 gather 往返在 ROCm（`:497`，`test_indexer_k_store_roundtrips_through_rocm_gather`）。另注：`:1752` 不是 V3.2 写入方，而是 **V4.0** 融合插入（`compress_ratio=4`）；V3.2 的 `indexer_k_quant_and_cache` 写入方测试在 `test_compressor_kv_cache.py:1425` |
| 3 | `indexer_sparse_logits` 的 opt-in 校验无测试 | `mla/sparse_indexer.py:105-153` 共 8 处 raise；测试目录对 `DeepseekV41SparseIndexer*` grep = 0 |
| 4 | sm90 上**两层候选筛选无端到端**用例 | 消费侧端到端被 SM100 门控（`test_mla_short_prefill_indexer.py:722-734`）；唯一端到端是 ROCm（`tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py:429`、`:463` 两个测试；`:320-418` 是其 `_run_layers` 辅助函数，不是测试） |
| 5 | TP row-shard 的**真实集合通信**无测试 | `test_indexer_tp_row_shard.py:67-74` 伪造 `all_gatherv` 并 monkeypatch `get_tp_group` |
| 6 | 无 sm90 CI lane 固定 `test_fused_indexer_q_rope_quant.py` 与 `test_compressor_kv_cache.py` | 这两个跑在 B200 lane（`kernels.yaml:235-245`）；sm90 lane 覆盖 `v1/attention`（`attention.yaml:5,8,15`）、`kernels/test_top_k_per_row.py`（`kernels.yaml:102-113`）、`model_executor`（`model_executor.yaml:5-6,29`） |

**一处写错的硬件常量**：`test_indexer_native_next_n.py:16` 写成 `NUM_SMS = 114  # H100 PCIe`，而 H200 是 132 个 SM（影响很小，因为平台被 monkeypatch 掉了；正确写法见 `test_indexer_deepseek_v4_slot_mapping.py:115`）。

**「叫得最响的那个反而不测 H200」**：唯一以 DeepSeek-V4.1 命名的 indexer 测试文件 `test_rocm_paged_mxfp4_indexer.py`，在 H200 上被**两层 skip 挡掉**，对 H200 没有任何贡献。

### 9.4 建议跑的最小验证命令

```bash
# 所有在 sm90 上真正执行的 indexer 测试（ROCm/SM100/CPU 参数自行 skip）
pytest -v -rs tests/kernels/test_fused_indexer_q_rope_quant.py \
  tests/model_executor/layers/test_mla_short_prefill_indexer.py \
  tests/v1/attention/test_indexer_{dcp_localize,deepseek_v4_slot_mapping,tp_row_shard,native_next_n}.py \
  tests/v1/attention/test_{kpool_indexer_block_sizes,sparse_indexer_decode_seq_lens,glm5next_indexer_block_sizes}.py \
  tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py \
  tests/models/glm5next/test_sparse_indexer_topk_dispatch.py \
  tests/kernels/test_compressor_kv_cache.py tests/kernels/test_top_k_per_row.py

# 反向验证：这些 lane 在 H200 上应当全 skip
pytest -v -rs tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py \
  tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py \
  tests/kernels/test_deepseek_v4_cpu_kernels.py
```

本调研自带的自检：

```powershell
$env:PYTHONPATH="E:\dsv41\.pylibs"
python E:\dsv41\vllm_0.32.0\indexer\indexer_torch_reference.py            # 18 个用例
python E:\dsv41\vllm_0.32.0\indexer\_work\make_diagrams.py --check-only   # 图对齐审计
```

---

**调研产物**：本文件（对外交付的单文件产物）；完整工作稿、可运行参考实现、逐条锚点校验脚本与
五份原始调研工作表在 `..\indexer\`（`INDEXER.md` / `INDEXER_diagram.md` /
`indexer_torch_reference.py` / `_work\`）。本文件的 §7 图块与 `make_diagrams.py --emit-section` 的输出**逐字节一致**
（`_work/diff_sec7.py` 断言）；正文的 290 条源码断言由 `_work/verify_claims.py` 校验，
编码 / 围栏 / 章节编号 / 行号范围与「同名文件必须带目录限定」由 `_work/verify_outputs.py` 校验，
164 条锚点由 `_work/spotcheck_anchors.py` 逐条比对被引用代码原文。
