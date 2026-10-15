# DSV4.1 mHC (multi-stream Hyper-Connection) — 调研与分类

> 目标硬件：**NVIDIA H200 / sm90（Hopper，compute capability 9.0）**
> 代码基线：**vllm-v0.32.0rc0**（vLLM v0.32.0rc0）
> 模型基线：DeepSeek-V4.1，40 层主干 + 3 层 MTP/dspark

---

## 0. 结论速览

1. **mHC 内没有量化算子，全场景只涉及两种 dtype。**
   - **BF16**：残差多流 `residual`、坍缩后的 `layer_input`、RMSNorm 权重、子层输出 `x`、aux 输出。
     （`hc_head` 输出也是 BF16，但 V4.1 已删除 learned `hc_head`，见第 3 条。）
   - **FP32**：全部门控系数（`post_mix` / `comb_mix` / `pre_mix`）、投影权重 `fn`、`hc_scale` / `hc_base`、GEMM 累加器与 sqrsum、split-k 中间缓冲。

2. **H200/sm90 上 mHC 只有一条主通路：TileLang。mHC 的投影 GEMM 子模块存在 DeepGEMM 分支 —— 但那是【分离路径】才有的分支。**
   落点先由 `mhc_fused_post_pre_split_config` 决定（`T ≤ 32 ∧ H % 128 == 0` → 融合）：
   - **融合路径（H200 decode，T ≤ 32）**：`_MHC_FUSED_TILELANG_KERNEL` 内部自算投影，**完全不查 `is_deep_gemm_supported()`** → **恒 TileLang FP32**（`tilelang.py:386-397`）
   - **分离路径（T > 32 或 H % 128 ≠ 0）**：显式调 `hc_prenorm_gemm`，此时才按 `is_deep_gemm_supported()` 二选一（`tilelang.py:411-432`）：
     - 可用：`tf32_hc_prenorm_gemm`（TF32 tensor-core，K-split）
     - 不可用：`hc_prenorm_gemm_tilelang`（**标量 FMA 循环**，无 tensor core）
   > ⚠️ 所以"sm90 默认 TF32"**只对 prefill 成立，对 decode 是错的**（详见 §2.2 / §7.6）。
   > 且 `is_deep_gemm_supported() = support_deep_gemm() ∧ VLLM_USE_DEEP_GEMM ∧ has_deep_gemm()`，**H200 上 `support_deep_gemm()` 为 True**，但环境变量或包缺失会使整条链为 False。

3. **V4.1 相对 V4 的三处结构性变化**：
   - **删掉了 learned `hc_head`**。末端坍缩改由最后一层 FFN 的 pre-mix 驱动 `hc_collapse_triton`。
   - **改成"延迟 / shifted" seam**：每个子层的坍缩用的是**上一层带进来的 `pre_mix`**，同时为本层算出 `next_pre_mix` 交给下一层；V4 则是"每个子层自己现算 pre_mix"的非延迟写法。这个差异会连带影响算子接口的选择（V4.1 必须换用延迟版算子）。
   - **每个子层拆成两套权重**：`hc_attn_fn` / `hc_ffn_fn` 各一套（V4 只有 `hc_attn_fn`）。

---

## 1. 术语与数学定义

- **`hc_mult`（HC 流数）**：残差被复制/维系成多少条并行"流"。DSV4.1 取 **`hc_mult = 4`**（`dsv41_part/config.json:128`）。
- **`mix_size` / `hc_mult3`（门控投影输出宽度）**：
  `mix_size = hc_mult * (hc_mult + 2)` = `4 * 6` = **24**。
  三段构成（顺序固定，见 `torch.py:108-118`）：
  | 切片 | 长度 | 含义 |
  |---|---|---|
  | `[0 : hc_mult]` | 4 | **pre**（坍缩门控 logits） |
  | `[hc_mult : 2*hc_mult]` | 4 | **post**（子层输出写入残差的系数 logits） |
  | `[2*hc_mult : ]` | 16 | **comb**（流间交叉混合矩阵 logits，`hc_mult × hc_mult`） |
- **`hc_scale`**：形状 `(3,)` FP32，三段各自的 gated logit 缩放（pre/post/comb）。
- **`hc_base`**：形状 `(mix_size,) = (24,)` FP32，三段各自的 logit 偏置。
- **门控公式**（`torch.py:108-123`，TileLang 在 `tilelang_kernels.py:196-241` 等价实现）：
  ```
  rms      = rsqrt( sum(x²)/ (hc_mult*hidden) + rms_eps )
  mixes    = (x_fp32 @ fnᵀ) * rms            # x = residual.flatten(1), [T, hc*hidden]
  pre_mix  = sigmoid(mixes[:, :hc]      * hc_scale[0] + hc_base[:hc])      + hc_pre_eps
  post_mix = sigmoid(mixes[:, hc:2hc]   * hc_scale[1] + hc_base[hc:2hc])   * hc_post_mult_value
  comb_log = mixes[:, 2hc:].view(T,hc,hc) * hc_scale[2] + hc_base[2hc:].view(1,hc,hc)
  comb_mix = softmax(comb_log, dim=-1) + hc_sinkhorn_eps
  comb_mix = comb_mix / (comb_mix.sum(-2,keepdim) + hc_sinkhorn_eps)
  repeat (sinkhorn_repeat - 1) times:
      comb_mix = comb_mix / (comb_mix.sum(-1,keepdim) + hc_sinkhorn_eps)
      comb_mix = comb_mix / (comb_mix.sum(-2,keepdim) + hc_sinkhorn_eps)
  ```
  → **Sinkhorn 归一化**（行列双向交替归一化）是 mHC 的核心，`sinkhorn_repeat = hc_sinkhorn_iters = 20`。
- **两端算子**：
  - **pre / collapse**：`layer_input = Σ_i pre_mix_i · residual_i`，随后可选融合 RMSNorm。
  - **post / expand**：`out_j = post_mix_j · x + Σ_i comb_mix_ij · residual_i`。
- **"delayed / shifted" seam（V4.1 特有）**：子层 *k* 的输入坍缩用子层 *k−1* 留下的 `pre_mix`；同时子层 *k* 算出自己的 `pre_mix` 交给 *k+1*。`fn` 投影同时产出 pre/post/comb 三段，因此"算门控"与"用门控"被错开一拍。

---

## 2. H200/sm90 运行时后端门控矩阵（关键）

### 2.1 四条加速路径的门控

```python 参考代码路径
# vllm/model_executor/layers/mhc.py:16-31
# vllm/models/deepseek_v41/nvidia/ops/mhc.py:44-56
# vllm/models/deepseek_v41/nvidia/ops/mhc.py:59-72
# vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py:16-28
```
**sm90 结论**：`supports_mhc_overlap=False` → `mhc_stream=None` → `fuse_mhc_all_reduce=False`
（`model.py:696-701`）→ `mhc_shifted_post_pre` 中 `stream is None` 且 `can_use_mega_mhc(...)` 为假
（`mega_mhc.py:38-44`）→ **唯一落点为 `mhc_fused_post_pre_delayed_tilelang`**（`ops/mhc.py:305`）。

### 2.1b ★ 三条门控的完整条件集

三条 gate 的**条件集互不相同**，而且**每一条都不只是 capability 判断** —— 这决定了它们在 H200 上是"恒假"还是"看环境才假"：

| 门控 | 完整条件（源文件） | H200/sm90 判定 |
|---|---|---|
| **`supports_mhc_overlap`**<br>`ops/mhc.py:44-56` | `family(100)` ∧ `is_deep_gemm_supported()` ∧ `hidden_size % 64 == 0` ∧ `0 < mix_size ≤ 32` ∧ `mix_size % 8 == 0` ∧ `not use_ubatching` | **恒 False**。第一项 `family(100)` 在 sm90 就是假；后三条对 `hc_mult=4`（`mix_size=24`）本来成立 |
| **`supports_mhc_all_reduce`**<br>`ops/mhc.py:59-72` | `tp_size == 4` ∧ `not enable_expert_parallel` ∧ `hidden_size == 5120` ∧ `hc_mult == 4` ∧ `ca_comm is not None` ∧ `ca_comm.mnnvl_lamport_ag_multicast_ptr` 非空 | **恒 False**。⚠️ 注意它**根本不判断 capability** —— 与 overlap 的"架构门"不同，这是**拓扑 + NVLink MNNVL 门**。在 H200 上即使 `tp=4`、`hidden=5120`、`hc_mult=4` 全部满足，也会因为 `mnnvl_lamport_ag_multicast_ptr` 为空而返回假（Lamport 多播指针是 **GB200 一类**硬件才有的能力）。**这是全表最容易读错的一条** |
| **`is_mega_mhc_supported(hidden_size, hc_mult)`**<br>`mega_mhc.py:16-28` | `is_deep_gemm_supported()` ∧ `family(100)` ∧ `hidden_size > 0` ∧ `hidden_size % 1024 == 0` ∧ `hc_mult == 4` ∧ `deep_gemm` 可导入且 `deep_gemm.mega_mhc` 可调用 | **恒 False**（`family(100)` 即假）。⚠️ 它带 **`@functools.cache`** → **每个进程只求值一次并锁存**，运行中改环境变量不会生效 |

**`is_deep_gemm_supported()` 自己还有两道门**（`deep_gemm.py:110-116`，同样是 `@functools.cache`）：
```
is_deep_gemm_supported() = support_deep_gemm() ∧ envs.VLLM_USE_DEEP_GEMM ∧ has_deep_gemm()
                           └─ platforms/cuda.py:688-695 ─┘   └─ 环境变量 ┘   └─ 已装 DeepGEMM 包 ┘
```
**H200 上 `support_deep_gemm()` 为 True**（它包含 `is_device_capability(90)`），但只要 `VLLM_USE_DEEP_GEMM=0`
或者**没装 DeepGEMM 包**，整条链就变 False —— 所以下面 §2.2 的"两个子分支"是**有条件的**，不是无条件成立。

### 2.2 sm90 上仍然存在的内部分叉：DeepGEMM prenorm GEMM

```python 参考代码路径
# vllm/models/deepseek_v41/nvidia/model.py:696
# vllm/model_executor/kernels/mhc/tilelang.py:33-71  (_hc_prenorm_gemm_outputs)
# vllm/platforms/cuda.py:688-695
```

> ⛔ **关键限定：下表只适用于【分离路径】。**
> C11 的落点先由 **决策 1**（§7.6）决定，而**融合路径恒用 TileLang、完全不看 DeepGEMM**：
> ```
> mhc_fused_post_pre_delayed_tilelang
>   ├─ 决策 1：mhc_fused_post_pre_split_config(T, H) 非 None ⇔ T ≤ 32 ∧ H % 128 == 0
>   │    ├─ 是 → 融合：_MHC_FUSED_TILELANG_KERNEL 内部自算投影
>   │    │        tilelang.py:386-397 —— 此处【没有】is_deep_gemm_supported() 判断
>   │    │        ⇒ 恒 TileLang FP32，n_splits = 8 固定
>   │    └─ 否 → 分离：显式调 hc_prenorm_gemm
>   │             tilelang.py:411-432 —— 此处【有】is_deep_gemm_supported() 判断
>   │             ⇒ 才轮到下面这张表的 A/B 选择
> ```
> **因此：H200 的 decode（T ≤ 32）是纯 TileLang FP32，prefill（T > 32）才可能走 DeepGEMM TF32。**
> 本文档 §7.6 与本处现已一致；若只读 §2 会误以为"sm90 默认就是 TF32"——那对 decode 是错的。

**分叉只发生在分离路径上 —— sm90 的两个子分支**：
| 子分支 | 条件 | 实现 | 数值精度 |
|---|---|---|---|
| **A. TF32 DeepGEMM** | `VLLM_USE_DEEP_GEMM=1`（默认）且已装 DeepGEMM 且其暴露 `tf32_hc_prenorm_gemm`（`deep_gemm.py:320, 946`） | `tf32_hc_prenorm_gemm`，K 维 split-k | **TF32 截断**（测试容差放宽到 `atol=1e-3, rtol=1e-3`，见 `test_mhc_kernels.py:294-295`） |
| **B. TileLang 标量 FMA** | DeepGEMM 不可用 / 未暴露该 API | `hc_prenorm_gemm_tilelang`（`tilelang_kernels.py:842-922`）：`x_val=cast(x[i_n,i_k],f32); acc[i_o]+=x_val*fn[out_idx,i_k]` 纯 FMA 循环，`n_splits=1` | **FP32 精确**（容差 `atol=1e-5, rtol=1e-4`） |

### 2.3 参考实现

| 层 | 文件 | 作用 |
|---|---|---|
| eager 参考 | `vllm/model_executor/kernels/mhc/torch.py` | `mhc_pre_torch` / `mhc_post_torch` / `mhc_pre_delayed_torch`，CUDA/CPU 的 fallback 与测试基准 |
| CPU | `vllm/model_executor/kernels/mhc/cpu.py` | 走 `ops.hc_pre_fused_cpu` 等 C++ op；**限制**：`n_splits==1`、`hc_pre_eps==hc_sinkhorn_eps`、`hc_post_mult_value==2.0` |
| XPU | `layers/mhc.py:623-635`（`torch.ops._xpu_C.mhc_post`）、`:748`（`_xpu_C.hc_head_fused`） | ⚠️ **XPU 不在 `torch.py` 里**，而是 `layers/mhc.py` 的 `forward_xpu` 分支 |
| ROCm / AITER | `vllm/model_executor/kernels/mhc/aiter.py`；`layers/mhc.py:406-415` 分派到 `mhc_fused_post_pre_delayed_rms_norm_aiter` | sm90 不可达（`is_rocm()` 为假）。**易被忽略的一层**，勿以为 TileLang 是唯一后端 |
| Triton | `vllm/model_executor/kernels/mhc/triton.py` | `hc_collapse_triton`（末端坍缩）、`hc_head_triton`（**V4.0 head 的 HIP 分支**，`layers/mhc.py:714`）、`mhc_pre_mix_triton`（AITER 专用） |
| TileLang | `vllm/model_executor/kernels/mhc/tilelang.py` + `tilelang_kernels.py` | **sm90 主力** |

---

## 3. 注册的算子清单（仅针对 DeepseekV4.1 实际涉及的部分）

### 3.1 `CustomOp` 层（`vllm/model_executor/layers/mhc.py`，按平台分派）

| CustomOp | 行 | CUDA 落点 | 说明 |
|---|---|---|---|
| `MHCPreDelayedOp` | 308-559 | `torch.ops.vllm.mhc_pre_delayed_tilelang` | 延迟 pre：用携带的 `pre_mix`，返回 `next_pre_mix` |
| `MHCPostOp` | 563-644 | `torch.ops.vllm.mhc_post_tilelang` | post：`out_j = post_j·x + Σ_i comb_ij·res_i` |
| `MHCFusedPostPreOp` | 768-1010 | `torch.ops.vllm.mhc_fused_post_pre_tilelang` | post+pre 融合（**非延迟**版，V4 用） |

### 3.2 `torch.ops.vllm.*` 注册表（`direct_register_custom_op`）

**TileLang（`tilelang.py`）共注册 6 个**：
`mhc_pre_delayed_tilelang`（`:1089`）、`mhc_fused_post_pre_delayed_tilelang`（`:1095`）、
`mhc_pre_tilelang`（`:1101`）、`mhc_post_tilelang`（`:1107`）、
`mhc_fused_post_pre_tilelang`（`:1114`）、`hc_head_fused_kernel_tilelang`（`:1121`）。
**Triton（`triton.py`）共注册 3 个**：`hc_head_triton`（`:330`）、`hc_collapse_triton`（`:337`）、`mhc_pre_mix_triton`（`:345`）。

> 完整清单（9 个 op）与"V4.1 是否可达"见 §10。


## 4. mHC 类型分类（编号域 `C1..C11`；**下文展开 V4.1 真实出现的 6 类：C2 / C3 / C5 / C7 / C8 / C11**）

编号 `C1..C11`，供结构图引用。本节展开 DeepseekV4.1 上**真实会被执行到的 6 类结构**：
C2（延迟 pre）、C3（post）、C5（post+pre 融合，主落点）、**C7（入口 broadcast）**、
**C8（Engram 注入，非 ED 分支特例）**、**C11（投影 GEMM，唯一有后端 fallback 的一类）**。
其余编号在 §7.6 对照表中作为**被替换/不适用的旧类**出现。

> **编号约定**：编号域是 `C1..C11`，但本文只展开 **V4.1 上真正会被执行到的 6 类**：
> C2 / C3 / C5 / C7 / C8 / C11。**C11 特别重要** —— 它是全子系统里
> **唯一拥有后端 fallback 的算子**（DeepGEMM TF32 ↔ TileLang FP32）。
> 其余编号（C1 / C4 / C6 / C9 / C10）属于 V4.0 或未落地的旧类，只在 §7.6 对照表里出现。

### C2 — `mhc_pre_delayed` / **shifted collapse**（V4.1 主 seam）

- **位置**：V4.1 `DeepseekV4DecoderLayer.forward` 的**每一个 attention 子层与 FFN 子层**（`deepseek_v41/nvidia/model.py:501-530, 548-561, 641-658`）。
- **代码**：`layers/mhc.py:308-559` → `kernels/mhc/tilelang.py:74-236`
- **语义**：用上一层携带的 `pre_mix` 坍缩本层输入，同时返回 `next_pre_mix`。
  ```python
  # tilelang_kernels.py:249-259
  if use_pre_mix_in:  pre_mix_shared[j] = pre_mix_in[i, j]
  elif save_pre_mix:  pre_mix_shared[j] = (j == 0) ? 1.0 : 0.0   # identity → 选流 0
  else:               pre_mix_shared[j] = sigmoid(...) + hc_pre_eps
  ```
  → 三种入口语义（**这是结构图必须区分的三态**）：
  | entry | 条件 | `pre_mix` 行为 |
  |---|---|---|
  | **identity** | `residual is None` 且 `x.dim()==3` | 未传 `pre_mix`，kernel 内选流 0（`use_pre_mix_in=False, save_pre_mix=True`） |
  | **broadcast** | `residual is None` 且 `x.dim()==2`（PP 非首/首 rank 衔接，或首层 embedding） | 走 C5，`fn_broadcast` + `rms_numel=H` |
  | **carried** | 常规层间传递 | 传入上一层 `pre_mix`（`use_pre_mix_in=True`） |
- **输入/输出**：同 C1，但 **多出** `pre_mix` 入参 `(T, hc_mult)` FP32 与 `next_pre_mix` 出参 `(T, hc_mult)` FP32；返回 4 元组 `(post, comb, layer_input, next_pre_mix)`。
- **sm90 落点**：`mhc_pre_delayed_tilelang` → `MHC_PRE_NORM_KERNEL`（`norm_weight` 非 None）或 `mhc_pre_big_fuse_tilelang`。

---

### C3 — `mhc_post` / 展开（CustomOp `MHCPostOp`）

- **位置**：V4.1 仅在 4 处：`_collapse()`（`model.py:1152`，最后一层收尾）、Engram 层（`model.py:540`）、MTP/dspark（`dspark.py:242`）、以及 `MHCPreDelayedOp._maybe_post`（`layers/mhc.py:334-344`）。
- **代码**：`layers/mhc.py:563-644` → `kernels/mhc/tilelang.py:791-811`
- **公式**：`out_j = post_layer_mix_j * x + Σ_i comb_res_mix_ij * residual_i`
- **输入**
  | 张量 | Shape | Dtype |
  |---|---|---|
  | `x`（子层输出） | `(T, H)` | **BF16** |
  | `residual` | `(T, hc_mult, H)` | **BF16** |
  | `post_layer_mix` | `(T, hc_mult, 1)`（kernel 内 `squeeze(-1)` → `(T,hc_mult)`） | **FP32** |
  | `comb_res_mix` | `(T, hc_mult, hc_mult)` | **FP32** |
- **输出**：`(T, hc_mult, H)` **BF16**
- **sm90 落点**：`_MHC_POST_TILELANG_KERNEL`（`tilelang_kernels.py:789-839`，`T.Kernel(n)`，FP32 fragment 累加后写回 BF16）

---

### C5 — **`mhc_fused_post_pre_delayed`**（★ V4.1 + sm90 的唯一主落点）

- **位置**：V4.1 **每个 decoder 子层边界**，经 `mhc_shifted_post_pre` 分派（`ops/mhc.py:305`）。
- **代码**：`kernels/mhc/tilelang.py:269-497`；kernel `tilelang_kernels.py` 的 `_MHC_FUSED_TILELANG_KERNEL` + `MHC_PRE_NORM_KERNEL`
- **语义**：把"上一层 post"折叠进"本层 pre 的 GEMM prologue"，残差流从寄存器直接喂投影，省掉一次 global 往返。
- **输入**

  | 张量 | Shape | Dtype |
  |---|---|---|
  | `x` | `(T, H)` | **BF16** |
  | `residual` | `(T, hc_mult, H)` | **BF16** |
  | `post_layer_mix` | `(T, hc_mult, 1)` | FP32 |
  | `comb_res_mix` | `(T, hc_mult, hc_mult)` | FP32 |
  | `fn` | `(mix_size, hc_mult*H)` → kernel 内 `view(24, hc_mult, H)` | FP32 |
  | `hc_scale` / `hc_base` | `(3,)` / `(24,)` | FP32 |
  | `pre_mix` 可选 | `(T, hc_mult)` | FP32 |
  | `norm_weight` 可选 | `(H,)` | **BF16** |
  | `capture_aux` | bool | — |

- **输出（6 元组）**

  | # | 名称 | Shape | Dtype |
  |---|---|---|---|
  | 0 | `residual_cur` | `(T, hc_mult, H)` | **BF16** |
  | 1 | `post_mix_cur` | `(T, hc_mult, 1)` | FP32 |
  | 2 | `comb_mix_cur` | `(T, hc_mult, hc_mult)` | FP32 |
  | 3 | `layer_input_cur`（可为已 RMSNorm） | `(T, H)` | **BF16** |
  | 4 | `next_pre_mix` | `(T, hc_mult)` | FP32 |
  | 5 | `aux`（`capture_aux` 时=流均值，否则 0 行） | `(T, H)` 或 `(0, H)` | **BF16** |

- **`capture_aux` 用途**：MTP/dspark 草稿模型要读取目标模型某一层的 hidden state。它被折叠进已有的坍缩循环（`tilelang_kernels.py:270-287`：`aux[i1_h] += xl[i_hc,i1_h]`，最后 `/ hc_mult`），**不额外读一遍残差**。
- **`split_mode` 两段式 epilogue**（Blackwell overlap 用；sm90 恒为 `"fused"`）：
  `"input"` 段先算坍缩（让输入尽快可用），`"stats"` 段算门控系数，两者在 `MHC_PRE_NORM_KERNEL.CompileKey.split_mode` 上分别特化（`warmup.py:39`、`ops/mhc.py:179-187`）。

---

### C7 — 入口 broadcast（模型入口 / PP 跨 rank 衔接）

- **位置 / 触发**：`DeepseekV4DecoderLayer.forward` 里 `x.dim() == 2` 的分支（`model.py:496-514`）。
  出现在两个场合：**首层的 embedding 输出**，以及 **PP 非首 rank 的跨 rank 衔接**。
- **代码**（`model.py:499-514`）：
  ```python
  assert self.hc_attn_fn_broadcast is not None
  residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()   # (T,H) → (T,hc,H)
  post_mix, res_mix, x, attn_pre = mhc_pre(
      residual, self.hc_attn_fn_broadcast, ...,
      x=x,                                    # ← 注意：把单流 x 也传进去当投影输入
      norm_weight=self.attn_norm.weight, norm_eps=...)
  ```
- **与 C2 的差别**：**没有 `pre_mix` 入参** —— 走的是 C2 的 `save_pre_mix=True` 分支，
  kernel 内取 `pre_mix = one_hot(0)`（等价于"选第 0 条流"），注释自述
  *"the identity pre-mix selects copy 0"*（`model.py:497-498`）。
- **`fn` 用的是派生权重**：`hc_attn_fn_broadcast` 不是检查点里的张量，而是运行时
  `hc_attn_fn.detach().view(-1, hc, H).sum(dim=1)` 算出来的（形状 `(24, H)`，`model.py:1418-1422`）。
  它的语义是"把 hc 条流的投影合成为单流投影"，从而让 `K = H` 而不是 `hc*H`（对比 §6.1 表）。
- **sm90 落点**：**纯 Python 的 `expand` + 常规 C2**。V4.0 那条专用的
  `mhc_pre_broadcast_tilelang`（且强制 `use_tilelang_fallback=False`）**V4.1 不再使用**。

### C8 — Engram 注入（非 ED 分支中的 mHC 结构）

```
residual ──▶ C3 (mhc_post_tilelang) ──▶ previous_post ⟨bf16⟩ (T,hc,H) ──┐
                                                                        ├─▶ Engram ──▶ residual' (T,hc,H) ──▶ C2
engram_hashes ⟨i32⟩ (T,2,24) ──[ :, layer_hash_index ]──▶ (T,24) ───────┘
```

- **位置**：`model.py:531-561`，`elif self.engram is not None and engram_hashes is not None` 分支。
  只在 **`engram_layer_ids = [1, 14]`** 两层出现。
- **为什么必须拆成三步**：源码注释写得很直白（`model.py:532-535`）——
  *Engram 注入发生在"上一子层的 post"与"本块的 pre"**之间**，作用在**完整的 hc 流**上，*
  *所以混合系数必须看到注入后的流；注入同时把 post 挡在了 pre-norm GEMM 的融合 prologue 之外。*
  → 这就是 Engram 层**不能用 C5 融合 kernel** 的根本原因（不是性能取舍，是数据依赖）。
- **三步顺序**（`model.py:540-561`）：
  1. `previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)` —— **C3 单独算**
  2. `residual = self.engram(previous_post, engram_hashes[:, layer_hash_index], engram_mask)`
  3. `mhc_pre(residual, self.hc_attn_fn, ..., pre_mix=pre_mix, norm_weight=...)` —— **C2 单独算**
- **形状**：`engram_hashes` 是 `(T, 2, 24)`，用 `[:, layer_hash_index]` 切出**本层那一份** `(T, 24)`
  （见 §6.1 的形状说明）。`Engram` 内部是一个融合 Triton kernel，它自己的 post/pre 是 Python 层编排，
  **不是** 调 C3/C2 的 kernel。
- **附带的 MoE 约束**：engram 层**前一层**必须自己收尾自己的 MoE（`model.py:536` 注释）。
- **aux 特例**：此分支下 `capture_previous_aux` 走的是 `previous_post.mean(dim=1)`（`model.py:541-542`），
  与 C5 分支里"折进坍缩循环"的做法**不同**。

### C11 — 投影 GEMM 本体（`hc_prenorm_gemm`，两类实现）

- **A. `tf32_hc_prenorm_gemm`**（`deep_gemm.py:932-954`，实现在外部 DeepGEMM 包）
  - 语义：`out = x.float() @ fn.T`（按 `num_split` 切 K），`sqrsum = x.float().square().sum(-1)`
  - 输入：`x (T, hc*H)` **BF16**、`fn (n_out, hc*H)` **FP32**；输出：`out (n_splits, T, n_out)` FP32、`sqrsum (n_splits, T)` FP32
  - 精度：**TF32**（10-bit 尾数）
- **B. `hc_prenorm_gemm_tilelang`**（`tilelang_kernels.py:842-922`）
  - `T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr)`；`n_thr=512, tile_n=12, n_splits=1`
  - 主体：`for it in serial(k_iters): x_val = cast(x[i_n,i_k], f32); for i_o in unroll(tile_n): acc[i_o] += x_val*fn[out_idx,i_k]`
  - `sqrsum` 只在 `i_t == 0` 的 tile 上累加
  - 精度：**FP32**（逐元素 FMA）
  - `n_splits == 1` 是硬要求（`assert x.shape[1] % n_splits == 0`、`(x.shape[1]//n_splits) % n_thr == 0`，`tilelang_kernels.py:1235-1236`）
  - `block_m` 变体（`hc_prenorm_gemm_block_m_tilelang`，926-1017）在 `num_tokens >= 1024` 时被 dispatch 选中（`tilelang_kernels.py:1147-1163`）；小 batch（`num_tokens < 128` 且 `hc_hidden_size % 1024 == 0`）改用 `n_thr=1024, tile_n=4`

C11 两条分支的数值精度不同，这是 sm90 上唯一真正影响结果的 fallback：

| 分支 | 实现 | 精度 | 测试容差 |
|---|---|---|---|
| A（sm90 默认） | tf32_hc_prenorm_gemm | TF32（10-bit 尾数） | 1e-3 |
| B | hc_prenorm_gemm_tilelang | FP32 标量 FMA | 1e-5 / 1e-4 |

---

## 5. 模块位置与作用（V4.1 主模型逐层视角）

```
┌─ DeepseekV4Model.forward  (deepseek_v41/nvidia/model.py:912)
│   embed_tokens → hidden_states (T, H) bf16            ← 单流
│   ...
│   ┌─ _run_layers(model.py:1093)  for idx in layer_ids:
│   │   ┌─ DeepseekV4DecoderLayer.forward (model.py:455)
│   │   │   ── ① 注意力边界 ─────────────────────────────
│   │   │   if residual is None:                      [首层 / PP 首块]
│   │   │       x.dim()==2 → C7 broadcast (model.py:499-514)
│   │   │       x.dim()==3 → C2 identity  (model.py:515-530)
│   │   │   elif engram is not None:                  [layer 1, 14] → C8
│   │   │       C3 post 单独算 (model.py:540)
│   │   │       engram 注入 (model.py:543)
│   │   │       C2 pre  单独算 (model.py:548)
│   │   │   else:                                    [常规层] → C5
│   │   │       mhc_shifted_post_pre(...) (model.py:565-584)
│   │   │   ── ② attn ── x = self.attn(positions, x, None)  (model.py:591)
│   │   │   ── ③ FFN 边界 ── mhc_shifted_post_pre(          → C5
│   │   │        pre_mix=attn_pre, hc_ffn_fn)   (model.py:599-617)
│   │   │   ── ④ FFN ── self.ffn(x, input_ids, ...)  (model.py:618-622)
│   │   └─ return x, residual, post_mix, res_mix, ffn_pre, previous_aux
│   └─ _collapse(model.py:1138)                     [最后一层]
│       C3 mhc_post_tilelang (model.py:1152)
│       (可选) aux = hidden_states.mean(dim=1) (model.py:1154)
│   末端: hc_collapse_triton(hidden_states, pre_mix) (model.py:1070)
│   self.norm(...)
└─

其中C11在模型中的位置：
model.py
  └─ mhc_shifted_post_pre()                    (ops/mhc.py:191)
       ├─ mhc_post_tilelang()          ← C3
       └─ mhc_fused_post_pre_delayed_tilelang() 或 mhc_pre_delayed_tilelang()
            └─ _hc_prenorm_gemm_outputs() / _MHC_FUSED_TILELANG_KERNEL
                 └─ C11  ──┬─ tf32_hc_prenorm_gemm        (DeepGEMM)
                           └─ hc_prenorm_gemm_tilelang    (TileLang)

```

**每个 decoder layer 携带的 mHC 参数**（`model.py:409-453`，全部 FP32、`requires_grad=False`）：

| 参数 | Shape | 张量大小 | 说明 |
|---|---|---|---|
| `hc_attn_fn` | `(24, 20480)` | 1.97 MB | attention pre/post/comb 投影 |
| `hc_ffn_fn` | `(24, 20480)` | 1.97 MB | FFN pre/post/comb 投影 |
| `hc_attn_base` | `(24,)` | 96 B | |
| `hc_ffn_base` | `(24,)` | 96 B | |
| `hc_attn_scale` | `(3,)` | 12 B | |
| `hc_ffn_scale` | `(3,)` | 12 B | |

**检查点证据**（`dsv41_part/model.safetensors.index.json`，共 96085 个 key）：
- 含 `hc_` 的 key：**258** = 43 × 6（`layers.0..39` 共 40 层 + `mtp.0..2` 共 3 层）
- `hc_head*` 出现次数：**0** → 确认 V4.1 无 learned hc_head
- 命名：`layers.{N}.hc_{attn,ffn}_{fn,base,scale}`、`mtp.{N}.hc_{attn,ffn}_{fn,base,scale}`
- 非 `layers.`/`mtp.` 前缀的 `hc*` key：**0**（即 vision tower / engram 无独立 mHC 权重）

**配置映射**（`dsv41_part/config.json` → `deepseek_v41/nvidia/model.py:342-345`）：

| config.json | 值 | model.py 属性 |
|---|---|---|
| `hc_mult` (line 128) | 4 | `self.hc_mult` |
| `hc_sinkhorn_iters` (line 129) | 20 | `self.hc_sinkhorn_iters` → `sinkhorn_repeat` |
| `hc_eps` (line 130) | 1e-06 | `self.hc_eps` → **同时**用作 `hc_pre_eps` 与 `hc_sinkhorn_eps` |
| —（硬编码） | 2.0 | `self.hc_post_alpha` (`model.py:345`) → `hc_post_mult_value` |
| `rms_norm_eps` (line 37) | 1e-20 | `rms_eps`（pre 的 rms）与 `norm_eps`（融合 RMSNorm） |

调用处实参顺序（`model.py:501-514`）：
```python
mhc_pre(residual, fn, hc_scale, hc_base,
        rms_eps=self.rms_norm_eps,      # 1e-20
        hc_pre_eps=self.hc_eps,         # 1e-6
        hc_sinkhorn_eps=self.hc_eps,    # 1e-6
        hc_post_mult_value=self.hc_post_alpha,  # 2.0
        sinkhorn_repeat=self.hc_sinkhorn_iters, # 20
        pre_mix=..., x=..., norm_weight=self.attn_norm.weight, norm_eps=self.attn_norm.variance_epsilon)
```

### MTP / dspark 草稿路径
`deepseek_v41/nvidia/dspark.py`：
- `self.hc_mult = config.hc_mult`（line 79）
- line 138 注释：`v4.1 has no learned hc_head: the hc copies are ...`
- line 227-228：`hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)`（`[T,H] → [T,hc,H]`）
- line 242：`mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)` → **C3**
- `combine_hidden_states(aux_hidden_states)`（line 161-166）：`main_norm(main_proj(concat of target aux hidden states))`；aux 由主模型的 `capture_aux` 产生（见 C5）
- **关键**：dspark 构造 `DeepseekV4DecoderLayer` 时**不传** `mhc_stream` / `fuse_mhc_all_reduce`（`dspark.py:126-133`）
  → 草稿层**恒走非 overlap 路径**：`mhc_pre_delayed_tilelang` + `mhc_shifted_post_pre(stream=None)`
- dspark 的 head（`markov_head` / `confidence_head` / `norm` / `lm_head`，`dspark.py:141-156, 343-349`）**不含 mHC**；
  v4 的 dspark 有 `hc_head_fn/_base/_scale`（`v4/nvidia/dspark.py:160-167, 274-278`），v4.1 无（参数路由表也删掉了这三个条目）

## 6. Shape / Dtype 总表

### 6.1 Shape 维度语义（`H=5120`, `hc_mult=4`, `T`=token 数, `hidden_block=gcd(1024,H)=1024`）

> **`hidden_block` 取值说明（容易算错，务必按 kernel 分）**：DSV4.1 **总是**传 `norm_weight`（`model.py:512, 528, 559, 579, 613` 等），
> 因此落点是 **`MHC_PRE_NORM_KERNEL`**，它在 `warmup.py:42` 绑定到
> **`mhc_pre_big_fuse_with_norm_tilelang`**，该函数用 **`hidden_block = math.gcd(1024, hidden_size)`**
> （`tilelang_kernels.py:337`）→ `H=5120` 时为 **1024**（Pipelined 2 段）。
> 另一处的 `hidden_block = math.gcd(512, hidden_size)` 在 **`tilelang_kernels.py:153`**，属
> `mhc_pre_big_fuse_tilelang`（**`norm_weight is None`** 的路径）—— **V4.1 从不走它**。
> 同理 `mhc_pre_big_fuse_broadcast_with_norm_tilelang` 也是 `gcd(1024, …)`（`:532`）。

| 张量 | Shape | 各维含义 |
|---|---|---|
| `residual` | `(T, hc_mult, H)` | token × HC 流 × 隐藏维 |
| `residual`（broadcast 入口） | `(T, H)` | token × 隐藏维（单流，未复制） |
| `residual`（broadcast 输出） | `(T, hc_mult, H)` | 复制后的多流（各流相等） |
| `fn` | `(mix_size, hc_mult*H)` = `(24, 20480)` | 门控输出 × 展平的多流输入 |
| `fn`（broadcast） | `(mix_size, H)` = `(24, 5120)` | 同上，但输入压缩为单流 |
| `fn`（fused kernel 内 view） | `(mix_size, hc_mult, H)` = `(24,4,5120)` | 便于按流分组求和 |
| `hc_fn`（hc_head） | `(hc_mult, hc_mult*H)` = `(4, 20480)` | head 门控数 × 展平输入 |
| `hc_scale` | `(3,)` | pre / post / comb 三段各一个缩放（hc_head 为 `(1,)`） |
| `hc_base` | `(mix_size,)` = `(24,)` | 每段逐元素偏置（hc_head 为 `(hc_mult,)`=`(4,)`） |
| `post_mix` / `post_layer_mix` | `(T, hc_mult, 1)` | token × 流（尾维为广播用的 1） |
| `comb_mix` / `res_mix` | `(T, hc_mult, hc_mult)` | token × 目标流 × 源流 |
| `pre_mix` / `next_pre_mix` / `attn_pre` / `ffn_pre` | `(T, hc_mult)` | token × 流 |
| `layer_input` / `x` | `(T, H)` | 坍缩后的单流子层输入 |
| `mixes` / `gemm_out_mul` | `(n_splits, T, mix_size)` | split-k 份 × token × 门控输出 |
| `sqrsum` / `gemm_out_sqrsum` | `(n_splits, T)` | split-k 份 × token 的 x² 行和 |
| `aux` | `(T, H)` 或 `(0, H)` | 残差流均值（`capture_aux=False` 时 0 行） |
| `norm_weight` | `(H,)` | RMSNorm 权重 |
| `engram_hashes` | **`(T, n_engram_layers, n_hash_cols)` = `(T, 2, 24)`** | INT32 索引 |

> **`engram_hashes` 形状说明（容易算错，三个维度各不相同）**：
> 生产端 `EngramHash.dummy_hashes()` 分配的是
> `(num_tokens, num_layers, (max_ngram - 1) * num_heads)`（`common/engram.py:512-516`），
> 其中 `n_hash_cols = (max_ngram_size - 1) * n_heads`（`engram.py:240`）。
> DSV4.1 配置 `engram_max_ngram_size=4`、`engram_n_heads=8`、`engram_layer_ids=[1,14]`（`config.json:131-141`）
> → **`(T, 2, 24)`**。传给 `Engram.forward` 的是**单层切片** `engram_hashes[:, k]` = **`(T, 24)`**，
> 与它的 docstring 契约 `hash_ids: [T, n_hash_cols]` 一致（`engram.py:1682`）。
> （`=12` 的算法是错的：`(4−1)×8 = 24`，不是 12。）

> **贯穿约定**（DSV4.1-Flash 实际取值）
> | 符号 | 含义 | 值 |
> |---|---|---|
> | `T` | token 数（动态） | — |
> | `hc` | `hc_mult`，HC 流数 | **4** |
> | `H` | `hidden_size` | **5120** |
> | `M` | `mix_size = hc*(hc+2)` | **24** (= pre 4 + post 4 + comb 16) |
> | `K` | 投影输入宽度 | `hc*H`=20480（C2/C5）或 `H`=5120（入口） |
> | `S` | split-k 份数 | `{1,4,16}`（DeepGEMM）或 `1`（TileLang） |
>
> dtype 图例：`⟨bf16⟩` = torch.bfloat16，`⟨f32⟩` = torch.float32，`⟨i32⟩` = torch.int32
> 所有门控系数、投影权重、累加器一律 `⟨f32⟩`；**整条通路无 FP8/FP4**。


## 7. 结构图（C11 → C2 → C3 → C5 逐层展开）

本节按 **C11 → C2 → C3 → C5** 的顺序逐层展开，每节给出「结构图 + 关键点 + 等价的纯 Torch 写法」，
最后一节（§7.6）说明 C3/C11 融合的那两个决策，以及全部结构的汇总对照表。

### 7.1 总览：4 个结构的位置与依赖

```
                          ┌──────────────────────────────────────────┐
      residual ⟨bf16⟩     │  C11  hc_prenorm_gemm                    │
      (T, hc, H)          │  x @ fnᵀ  →  mixes ⟨f32⟩ (S,T,M)         │
          │               │  x².sum   →  sqrsum⟨f32⟩ (S,T)          │
          │               │  ┌─ DeepGEMM TF32 (分离路径) ─┐          │
          └──flatten(1)──▶│  │  tf32_hc_prenorm_gemm      │         │
          (T, K)          │  └─ TileLang FP32 标量 FMA ────┘         │
                          └────────────────┬─────────────────────────┘
                                           │ mixes, sqrsum
                                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  C2  mhc_pre_delayed   （坍缩 + 生成三组门控）                 │
        │  ┌─ thread 0..31  ──▶ post_mix ⟨f32⟩(T,hc,1)                  │
        │  │                    comb_mix ⟨f32⟩(T,hc,hc)  [Sinkhorn×20]  │
        │  └─ thread 32..95 ──▶ layer_input ⟨bf16⟩(T,H)                 │
        │                        Σ_i pre_mix_i · residual_i (+RMSNorm)  │
        │  同时产出 next_pre_mix ⟨f32⟩(T,hc)  ← 接力给下一子层           │
        └──────────────────────────────────────────────────────────────┘
                          │ layer_input ⟨bf16⟩(T,H)
                          ▼
                    ┌───────────┐
                    │ attention │  /  FFN
                    └─────┬─────┘
                          │ x ⟨bf16⟩(T,H)
                          ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  C3  mhc_post                                                 │
        │  out_j = post_j ⊙ x  +  Σ_i comb[i,j] · res_i                 │
        │  (T,hc,H) ⟨bf16⟩ ◀── 内部 ⟨f32⟩ 累加                          │
        └──────────────────────────────────────────────────────────────┘
                          │ residual ⟨bf16⟩(T,hc,H)
                          ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  C5  mhc_fused_post_pre_delayed                              │
        │      = C3 + C11 + C2  按 2 个独立决策拼装                    │
        │      ┌ C3 ──▶ C11 ──▶ C2 ┐                                   │
        │      └── 融合点 ──┘      │      三者地位平等，从不被融合掉   │
        │  决策1  C3 融合/分离  ── 只抹掉 C3→C11 的边界                │
        │  决策2  C11 后端 ── 融合路径恒 TileLang；分离路径 → DeepGEMM │
        └──────────────────────────────────────────────────────────────┘
```
**末端特例**：最后一层之后是 `C3 → hc_collapse_triton → RMSNorm`。
`hc_collapse_triton` **就是 C2 的坍缩那一步**（同一个式子，换了 `pre_mix` 的来源），**不是 C3**。

### 7.2 C11 — 投影 GEMM `hc_prenorm_gemm`

#### 7.2.1 结构图

```
       x ⟨bf16⟩ (T, K)                    fn ⟨f32⟩ (M, K)
     = residual.flatten(1)               （hc_attn_fn / hc_ffn_fn
       K = hc*H = 20480                   / hc_attn_fn_broadcast[K=H]）
            │                                    │
            │   ┌──────────── split-k over K ─────┴────────────┐
            │   │                                              │
            ▼   ▼                                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  branch A (分离路径) DeepGEMM  tf32_hc_prenorm_gemm             │
   │    x_f32 @ fnᵀ        ── TF32 tensor core ──▶ mixes (S, T, M) f32 │
   │    x_f32.square().sum(-1) ────────────────▶ sqrsum (S, T)    f32 │
   │    S = compute_mhc_pre_num_splits(K, T) ∈ {1, 4, 16}            │
   ├─────────────────────────────────────────────────────────────────┤
   │  branch B (DeepGEMM 不可用或融合路径): hc_prenorm_gemm_tilelang │
   │    grid = (T, ceil(M/tile_n)=2, n_splits=1), threads=512,       │
   │    tile_n=12  →  纯 FP32 标量 FMA 循环，无 tensor core          │
   │    S 恒为 1                                                     │
   └─────────────────────────────────────────────────────────────────┘
            │                          │
            ▼                          ▼
      mixes (S,T,M) ⟨f32⟩       sqrsum (S,T) ⟨f32⟩
            │                          │
            └──────────┬───────────────┘
                       ▼
              C2 的 epilogue 入口
              （S 份在 epilogue 里先求和归约：Σ_s mixes[s,i,:]）
```

#### 7.2.2 等价的纯 Torch 写法

```python
# ---- C11 投影 ----
def c11_prenorm_gemm(residual, fn, *, rms_numel=None, x=None):
    """
    residual : (T, hc, H) bf16      -- C2/C5 常规路径
    x        : (T, K)    bf16       -- 可选；入口 broadcast 时传 x=(T,H)
    fn       : (M, K)    f32
    returns  : mixes (T, M) f32,  sqrsum (T,) f32,  rms_numel int
    """
    T, hc, H = residual.shape
    if x is None:
        x = residual.flatten(1)                       # (T, hc*H)
    if rms_numel is None:
        rms_numel = x.shape[1]                        # hc*H，入口为 H

    x = x.float()                                     # ← 上游强制转 f32
    mixes  = x @ fn.t()                               # (T, M)
    sqrsum = x.square().sum(dim=-1)                   # (T,)  只对 K 求和

    # ---- split-k 归约等价于先求和；显式写出来是为了对齐 kernel 的两段结构 ----
    # mixes  = mixes.sum(dim=0)     # 若上游给的是 (S,T,M)
    # sqrsum = sqrsum.sum(dim=0)    # 若上游给的是 (S,T)
    return mixes, sqrsum, rms_numel
```

### 7.3 C2 — `mhc_pre_delayed`（坍缩 + 三组门控）

#### 7.3.1 结构图

```
  residual ⟨bf16⟩ (T,hc,H)   mixes ⟨f32⟩(T,M)   sqrsum⟨f32⟩(T,)   pre_mix_in ⟨f32⟩(T,hc)
         │                        │                  │                   │
         │                        │                  │                   │
  ┌──────┴────────────────────────┴──────────────────┴───────────────────┴──────┐
  │  T.Kernel(num_tokens, threads=96)   ← 96 线程按角色切成两半                  │
  │                                                                             │
  │  ┌── 全部 96 线程：RMS 归一化 mixes ────────────────────────────────────┐   │
  │  │   rms[0] = rsqrt( Σ_s sqrsum[s,i] / rms_numel + rms_eps )   ⟨f32⟩    │   │
  │  │   mixes[j] = ( Σ_s mixes[s,i,j] ) * rms[0]                j∈[0,24)  │   │
  │  │   ──▶ mixes_shared ⟨f32⟩(24,)   ← 共享内存，跨 warp 传递的接缝       │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  │                                                                             │
  │  ┌── thread 0..31 分支 (`_is_mix_thread`) ─────────────────────────────┐   │
  │  │                                                                     │   │
  │  │  ① 门控 logits (M=24 拆成三段)                                       │   │
  │  │     pre_logits[j]  = mixes_shared[j]           * hc_scale[0] + hc_base[j]        j∈[0,4) │
  │  │     post_logits[j] = mixes_shared[j+hc]        * hc_scale[1] + hc_base[j+hc]     j∈[0,4) │
  │  │     comb_logits[j,k]= mixes_shared[j*hc+k+2*hc]* hc_scale[2] + hc_base[j*hc+k+2*hc]     │
  │  │                                                                     │   │
  │  │  ② 输出系数                                                          │   │
  │  │     post_mix[i,j] = sigmoid(post_logits[j]) * hc_post_mult_value     │   │
  │  │                       └─ hc_post_mult_value = 2.0（硬编码 hc_post_alpha）│   │
  │  │                                                                     │   │
  │  │  ③ Sinkhorn 归一化（repeat = hc_sinkhorn_iters = 20）                 │   │
  │  │     comb = softmax(comb_logits, dim=-1) + hc_sinkhorn_eps            │   │
  │  │     comb = comb / (comb.sum(-2) + hc_sinkhorn_eps)                   │   │
  │  │     for _ in range(repeat - 1):                                      │   │
  │  │         comb = comb / (comb.sum(-1) + hc_sinkhorn_eps)   # 行归一      │   │
  │  │         comb = comb / (comb.sum(-2) + hc_sinkhorn_eps)   # 列归一      │   │
  │  │                                                                     │   │
  │  │  ④ next_pre_mix[i,j] = sigmoid(pre_logits[j] * ... ) + hc_pre_eps     │   │
  │  │       注意：pre_logits 这里用的是 mixes_shared[j]（已含 rms）          │   │
  │  │       save_pre_mix=False 时才算；V4.1 恒为 True                       │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │              │ post_mix ⟨f32⟩(T,hc)   comb_mix ⟨f32⟩(T,hc,hc)               │
  │              │ next_pre_mix ⟨f32⟩(T,hc)                                      │
  │              ▼                                                               │
  │  ┌── thread 32..95 分支 (`_is_collapse_thread`) ───────────────────────┐   │
  │  │                                                                     │   │
  │  │  ⑤ 选择坍缩系数 pre_mix_shared                                        │   │
  │  │     if use_pre_mix_in:  pre_mix_shared[j] = pre_mix_in[i, j]         │   │
  │  │     elif save_pre_mix:  pre_mix_shared[j] = (j==0) ? 1.0 : 0.0  ← identity │
  │  │     else:               sigmoid(pre_logits[j]) + hc_pre_eps          │   │
  │  │                                                                     │   │
  │  │  ⑥ 沿 hc 维加权求和（hidden_block = gcd(1024,H) = 1024，Pipelined 2 段）│  │
  │  │     for i0_h in Pipelined(H/hidden_block):                           │   │
  │  │         xs = residual[i, :, i0_h*hb : (i0_h+1)*hb]   ⟨bf16⟩→⟨f32⟩     │   │
  │  │         ol[·] = Σ_{i_hc} pre_mix_shared[i_hc] * xs[i_hc, ·]   ⟨f32⟩   │   │
  │  │         (可选 write_aux) aux[·] = Σ_{i_hc} xs[i_hc,·] / hc            │   │
  │  │         layer_input[i, i0_h*hb+·] = bf16(ol[·])                       │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────┘
        │                    │              │                │
        ▼                    ▼              ▼                ▼
  post_mix            comb_mix       layer_input        next_pre_mix
  (T,hc,1)⟨f32⟩     (T,hc,hc)⟨f32⟩  (T,H)⟨bf16⟩        (T,hc)⟨f32⟩
                                       │                    │
                                  → attention/FFN       → 下一子层
                                                        （延迟一拍）
```

#### 7.3.2 等价的纯 Torch 写法

```python
# ---- C2 坍缩 + 门控（对应 mhc_pre_delayed_torch） ----
def c2_pre_delayed(residual, fn, hc_scale, hc_base,
                   rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                   hc_post_mult_value, sinkhorn_repeat,
                   pre_mix=None, x=None):
    """
    residual : (T, hc, H) bf16
    fn       : (M, hc*H)  f32
    hc_scale : (3,)       f32          pre / post / comb
    hc_base  : (M,)       f32
    pre_mix  : (T, hc)    f32 or None  None → identity（选流 0）
    x        : (T, K)     bf16 or None 入口 broadcast 时给 (T, H)
    returns  : post (T,hc,1) f32, comb (T,hc,hc) f32,
               layer_input (T,H) bf16, next_pre (T,hc) f32
    """
    hc = residual.shape[1]

    # ---------- ① C11 投影 ----------
    x = (residual.flatten(1) if x is None else x).float()      # (T, K)
    mixes = x @ fn.t()                                         # (T, M)
    # ---------- ② RMS 归一化（split-k 已归约） ----------
    mixes = mixes * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)

    # ---------- ③ 三段门控 logits ----------
    pre_logits  = mixes[:, :hc]            * hc_scale[0] + hc_base[:hc]
    post_logits = mixes[:, hc:2 * hc]      * hc_scale[1] + hc_base[hc:2 * hc]
    comb_logits = mixes[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] \
                  + hc_base[2 * hc:].view(1, hc, hc)

    # ---------- ④ 输出系数 ----------
    post = torch.sigmoid(post_logits) * hc_post_mult_value      # (T, hc)

    # ---------- ⑤ Sinkhorn（这是 mHC 的核心特征）----------
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)   # 行
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)   # 列

    # ---------- ⑥ 坍缩（hc_collapse_triton 就是这一行）----------
    layer_input = (
        residual[:, 0]                                          # identity 特例
        if pre_mix is None
        else (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    )
    # 可选融合 RMSNorm（norm_weight 非 None 时，C2 直接吐归一化后的 layer_input）
    # layer_input = rms_norm(layer_input, norm_weight, norm_eps)

    # ---------- ⑦ 接力给下一子层 ----------
    next_pre = torch.sigmoid(pre_logits) + hc_pre_eps           # (T, hc)

    return post.unsqueeze(-1), comb, layer_input, next_pre
```

**`hc_collapse_triton` 的等价写法（末端坍缩，就是上面 ⑥ 的 else 分支）**

```python
def hc_collapse(hidden_states, pre_mix):
    """hidden_states: (T, hc, H) bf16   pre_mix: (T, hc) f32  → (T, H) bf16"""
    return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(dim=1).to(hidden_states.dtype)
```

两个 bitwise 不变量（可用来验证聚合语义，见 `test_mhc_kernels.py:119-137`）：
- `pre_mix = [0,0,0,1]` → 输出**恰好等于** `hidden_states[:, 3, :]`（`atol=rtol=0`）
- `pre_mix = [1,1,1,1]` 且 `x[:,1]=-x[:,0]`, `x[:,3]=-x[:,2]` → 输出**恰好全零**

---

### 7.4 C3 — `mhc_post`

#### 7.4.1 结构图

```
      x ⟨bf16⟩(T,H)        residual ⟨bf16⟩(T,hc,H)   post ⟨f32⟩(T,hc,1)   comb ⟨f32⟩(T,hc,hc)
           │                        │                        │                    │
           │                        │                        │                    │
  ┌────────┴────────────────────────┴────────────────────────┴────────────────────┴────┐
  │  T.Kernel(num_tokens, threads=n_thr)      ← 每个 token 一个 block，token 间无耦合  │
  │                                                                                    │
  │   把 (hc,hc) 与 (hc,) 系数一次性搬进 fragment ⟨f32⟩：                              │
  │     a_local = comb[i_n, :, :] ⟨f32⟩(hc,hc)      ← comb_res_mix                     │
  │     c_local = post[i_n, :]    ⟨f32⟩(hc,)        ← post_layer_mix.squeeze(-1)        │
  │                                                                                    │
  │   for i0_h in Serial( H / h_blk ):        ← h_blk = gcd(1024, H) = 1024，逐块       │
  │     ┌──────────────────────────────────────────────────────────────────┐           │
  │     │  b_shared = residual[i_n, :, blk] ⟨bf16⟩(hc,h_blk)               │           │
  │     │  d_shared = x[i_n, blk]           ⟨bf16⟩(h_blk)                  │           │
  │     │      ↓ 搬进 ⟨f32⟩ fragment                                        │           │
  │     │                                                                │           │
  │     │  ┌─ 每个 (i_hco, i1_h) 输出元素 ────────────────────────────┐   │           │
  │     │  │  x_local[i_hco,i1_h]  = c_local[i_hco] * d_local[i1_h]    │   │           │
  │     │  │                          └────── post_j ⊙ x ──────┘        │   │           │
  │     │  │  for i_hci in vectorized(hc):                             │   │           │
  │     │  │      x_local[i_hco,i1_h] += a_local[i_hci,i_hco]          │   │           │
  │     │  │                             * b_local[i_hci,i1_h]         │   │           │
  │     │  │        └── Σ_i comb[i, j] · res_i     ★ 索引是 [i_hci, i_hco] │           │
  │     │  └───────────────────────────────────────────────────────────┘   │           │
  │     │                                                                │           │
  │     │  out[i_n, :, blk] = bf16(x_local)     ← 只在最后一次舍入到 bf16  │           │
  │     └──────────────────────────────────────────────────────────────────┘           │
  └────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          out ⟨bf16⟩ (T, hc, H)
```

#### 7.4.2 关键点
- **`out` 与 `residual` 同形** `(T,hc,H)` —— C3 是"流数不变"的算子（对比 C2 是坍缩、入口 expand 是扩张）。
- **`comb` 的索引方向**：kernel 里是 `a[i_hci, i_hco] * b[i_hci, ...]`，即输出流 `j=i_hco` 由**第一维是源流**的 `comb[源, 目标]` 决定。这对应 torch 的 `comb_res_mix.mT`（见下）。**画图/实现时最容易搞反的地方。**
- 内部 `x_local` / `b_local` / `a_local` / `c_local` **全是 `⟨f32⟩`**，只在写回时 `bf16()` 舍入一次。
- `post` 传进来是 `(T,hc,1)`，kernel 入口 `squeeze(-1)` 成 `(T,hc)`（`tilelang.py:805`）。

#### 7.4.3 等价的纯 Torch 写法

```python
# ---- C3 post（对应 mhc_post_torch，vLLM 官方参考） ----
def c3_post(residual, x, post_layer_mix, comb_res_mix):
    """
    x              : (T, H)       bf16   -- 子层输出
    residual       : (T, hc, H)   bf16
    post_layer_mix : (T, hc, 1)   f32    -- 注意尾维是 1
    comb_res_mix   : (T, hc, hc)  f32
    returns        : (T, hc, H)   bf16
    """
    mixed = torch.einsum("...ij,...ih->...jh",
                         comb_res_mix.to(torch.float32),   # ← [源 i, 目标 j]
                         residual.to(torch.float32))
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed + post_term).to(residual.dtype)
```

**最直观的写法**（数学等价、索引方向一眼可见）：

```python
def c3_post_matmul(residual, x, post_layer_mix, comb_res_mix):
    """用 mT 把 [源,目标] 转成 [目标,源]，再 bmm —— 这就是 kernel 的 a[i_hci,i_hco]。"""
    term2 = torch.bmm(comb_res_mix.mT.float(), residual.float())        # (T, hc, H)
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
```

> 这个 `mT` 写法正是 vLLM 测试里的 `mhc_post_ref`
> （`test_mhc_kernels.py:916-924`，注释指向 tilelang 官方 deepseek_mhc 示例）。

**广播语义提示**：`post_layer_mix` 的 `(T,hc,1)` 与 `x.unsqueeze(-2)` 的 `(T,1,H)` 相乘 → `(T,hc,H)`，
即"同一个 `x` 按每个目标流的 `post_j` 缩放后加到该流上"。

---

### 7.5 C5 — `mhc_fused_post_pre_delayed`（融合 C3 + C11 + C2；C2 恒定参与）

#### 7.5.1 结构图

```
x          residual      post         comb          fn
bf16(T,H)  bf16(T,hc,H)  f32(T,hc,1)  f32(T,hc,hc)  f32(M,K)
    │            │            │             │           │
    └────────────┴────────────┴─────────┘               │
                                        ▼               │
     ┌────────────────────────────────────────────────────────────────────┐
     │ 决策 1 = C3 的实现（融合判据 mhc_fused_post_pre_split_config）     │
     │   T > 32          → None  ─┐                                       │
     │   H % 128 != 0    → None  ─┴─▶ 退回「分离」                        │
     │   T <  16 → (tile_n, n_splits, n_thr) = (2, 8, 128)                │
     │   T >= 16 → (6, 8, 128)                                            │
     │   再收缩：mix_size % tile_n == 0，H % (n_splits*n_thr) == 0        │
     └──────────────────────────────────┬─────────────────────────────────┘
                                        │ 决策 1
                  ┌─────────────────────┴─────────────────────────────────────────────────────────────────┐
                                        ▼                                                                 ▼
                  ┌────────────────────────────────────────────┐                    ┌────────────────────────────────────────────┐
                  │ _MHC_FUSED_TILELANG_KERNEL                 │                    │ ① C3   _MHC_POST_TILELANG_KERNEL           │
                  │ 把 C3 的 post 折叠进投影的                 │                    │       residual_cur  bf16 (T,hc,H)          │
                  │ prologue：残差流留在寄存器里               │                    │ ② C11  决策 2 在此生效：                   │
                  │ 直接喂 GEMM，省一次 global                 │                    │       有 DeepGEMM → TF32  S∈{1,4,16}       │
                  │ 往返。                                     │                    │       否则        → TileLang FP32  S=1     │
                  │                                            │                    │       mixes   f32 (S,T,24)                 │
                  │ ★ 此 kernel 内部自算投影，                 │                    │       sqrsum  f32 (S,T)                    │
                  │   不调用 tf32_hc_prenorm_gemm              │                    │                                            │
                  │   ⇒ 决策 2 无选择余地，                    │                    │                                            │
                  │     恒为 TileLang FP32。                   │                    │                                            │
                  └────────────────────────────────────────────┘                    └────────────────────────────────────────────┘
                  │                     │                                                                 │
                  └─────────────────────┴─────────────────────────────────────────────────────────────────┘
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ C2 = 恒定尾段：两条路径共用，从不被融合掉                            │
    │                                                                      │
    │ Ⅰ  RMS 归一化 mixes_shared（全 96 线程）                             │
    │      rms = rsqrt(Σ_s sqrsum[s,i] / rms_numel + rms_eps)              │
    │      mixes[j] = (Σ_s mixes[s,i,j]) * rms[0]                          │
    │                                                                      │
    │ Ⅱ  gate（thread 0..31）← hc_scale/hc_base 在此才用                   │
    │      pre  = sigmoid(mixes[j]    *hc_scale[0]+hc_base[j])             │
    │      post = sigmoid(mixes[j+hc] *hc_scale[1]+hc_base[j+hc])*2.0      │
    │      cm   =         mixes[j*hc+k+2hc]*hc_scale[2]+hc_base[..]        │
    │ Ⅲ  Sinkhorn 归一化（thread 0..31，×20 迭代）                         │
    │                                                                      │
    │ Ⅳ  pre_mix（thread 32..95）                                          │
    │      pre_mix_in（延迟携带）或 one_hot(0)（entry=选流0）              │
    │ Ⅴ  坍缩 Σ_i pre_mix_i · residual_cur_i → bf16                        │
    │      （+ 融合 RMSNorm；+ capture_aux 时顺带产出 aux）                │
    └──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼  返回 6 个张量

    #  名称              来源                          shape / dtype
    -- ----------------- ----------------------------- ----------------------
    0  residual_cur      C3 的 per-token post           (T,hc,H) bf16
    1  post_mix_cur      C2 gate logits                (T,hc,1) f32
    2  comb_mix_cur      C2 gate logits + Sinkhorn x20 (T,hc,hc) f32
    3  layer_input       C2 坍缩（可融合 RMSNorm）            (T,H) bf16
    4  next_pre_mix      C2 gate logits（延迟接力）          (T,hc) f32
    5  aux               C2 坍缩顺带产出                     (T,H) bf16 / (0,H)
```

#### 7.5.2 关键点

- **C5 不是新数学**，就是 `C3 → C11 → C2` 三步串联；融合的收益是把 C3 产出的 `residual_cur` 留在寄存器里直接当投影的左操作数，省一次 global 写读。
- **三个算子地位平等**：C5 的 kernel 把三件事按**三套独立 tile schedule** 融合 ——
  C11 是 M-tile × K-split 的 GEMM 网格；C3 是 per-token × hidden-block 串行循环；
  C2 是 per-token block 内按 `_is_mix_thread` / `_is_collapse_thread` 切两半。
  三套调度正是"三个算子"的标志。
- **`mhc_fused_post_pre_split_config` 返回的 `(tile_n, n_splits)` 是 C11 的调度参数**
  （切 M 与切 K），不是 C2/C3 的 —— 融合决策本身就是在做 C11 的 schedule 选择。
- **返回值有 6 个**，逐项对应 3 个算子的哪一段：

  | # | 输出 | 来源 | shape / dtype |
  |---|---|---|---|
  | 0 | `residual_cur` | **C3** 的 per-token post | `(T, hc, H)` bf16 |
  | 1 | `post_mix_cur` | **C2** 的 gate logits | `(T, hc, 1)` f32 |
  | 2 | `comb_mix_cur` | **C2** 的 gate logits + Sinkhorn×20 | `(T, hc, hc)` f32 |
  | 3 | `layer_input_cur` | **C2** 的坍缩（可已融合 RMSNorm） | `(T, H)` bf16 |
  | 4 | `next_pre_mix` | **C2** 的 gate logits（延迟接力） | `(T, hc)` f32 |
  | 5 | `aux` | **C2** 的坍缩循环附带产出 | `(T, H)` 或 `(0, H)` bf16 |

  → **C3 只贡献第 0 项**；C11 不直接产生输出（它只喂 C2）；C2 贡献第 1~5 项。
- `aux` **不额外读一遍残差** —— 它被折进已有的坍缩循环（`tilelang_kernels.py:270-287`：`aux += xs[i_hc]`，最后 `/ hc`）。
- **注意 `fn` 在融合 kernel 里被 view 成 `(24, hc, H)`**，与 C11 里的 `(24, hc*H)` 是同一块内存。

---

### 7.6 ★ C5 的两个独立决策（本子系统最反直觉的一点）

```
                     ┌──────────── C5 内部 ─────────────┐
residual, x  ──▶ C3 ──▶ C11 ──▶ C2 ──▶ 6 个输出
                     └──────────────────────────────────┘
                      └─融合点─┘
                 （C3 的 post 折进 C11 的 prologue）

决策 1: C3 实现      融合  (T ≤ 32 且 H % 128 == 0)
                     分离  (否则退回独立 post kernel)
                           ↑ 只抹掉 C3→C11 的边界

决策 2: C11 后端     融合路径 → 恒用 TileLang（hc_prenorm_gemm）
                     分离路径 → DeepGEMM 可用则 TF32，否则 TileLang
                           ↑ 独立决定
```

**决策 1 与决策 2 并不正交 —— 决策 1 会关掉决策 2 的一个选项**，这是本图最反直觉的一点：

| 决策 1 | C11 实现 | 是否可用 DeepGEMM TF32 | split-k |
|---|---|---|---|
| **融合**（`T ≤ 32`） | `_MHC_FUSED_TILELANG_KERNEL` 内部自算 | **否，恒 TileLang** | `n_splits = 8`（固定 tile 配置） |
| **分离**（`T > 32` 或 `H % 128 != 0`） | `hc_prenorm_gemm` 显式调用 | 是（`is_deep_gemm_supported()`） | `{1, 4, 16}` 桶化 |

证据：融合分支 `tilelang.py:386-397` **没有** `is_deep_gemm_supported()` 判断；
分离分支 `tilelang.py:411-432` 有。

**实际含义**：小 batch（decode，T≤32）走 **TileLang FP32**，大 batch（prefill，T>32）走 **DeepGEMM TF32**。
→ **精度特征随路径翻转**（FP32 精确 ↔ TF32 10-bit 尾数），不只是性能差异。这也解释了测试里
`use_deep_gemm` 那条参数化只对分离形状有意义。

---

### 7.7 汇总：一句话对照表

| 结构 | 数学 | Torch 一行核心 | 输入 shape | 输出 shape | 有后端 fallback？ |
|---|---|---|---|---|---|
| **C11** | `x @ fnᵀ`, `x².sum` | `x.float() @ fn.t()` | `(T,K)` bf16 | `(S,T,M)`+`(S,T)` f32 | ✅ **唯一有**（DeepGEMM TF32 ↔ TileLang FP32） |
| **C2** | Sinkhorn 门控 + 加权坍缩 | `(pre.unsqueeze(-1) * res.float()).sum(1)` | `(T,hc,H)` bf16 | `(T,hc,1)`,`(T,hc,hc)` f32 + `(T,H)` bf16 | ❌ |
| **C3** | 流间混合 + 输出注入 | `torch.bmm(comb.mT.float(), res.float()) + post*x` | `(T,hc,H)` bf16 + `(T,H)` bf16 | `(T,hc,H)` bf16 | ❌ |
| **C5** | C3 ∘ C11 ∘ C2（C2 恒定参与） | `c5 = c2(c3(...))` | 同上 | 6 元组 | 经 C11 间接拥有 |
| **末端 collapse** | C2 的坍缩一步 | `(pre.unsqueeze(-1) * hs.float()).sum(1)` | `(T,hc,H)` bf16 | `(T,H)` bf16 | ❌ |

## 8. 单元测试覆盖情况（C2 / C3 / C5 / C11）

### 8.1 测试资产总览

测试目录：`tests/kernels/mhc/`

| 文件 | 行数 | 内容 | 是否需要 GPU |
|---|---|---|---|
| `test_mhc_kernels.py` | 1760 | **主力**：逐个结构的数值正确性 UT + 层间集成 | 是（CUDA/ROCm） |
| `test_mhc_jit_warmup.py` | 325 | 5 个 TileLang kernel 的 `dispatch` / `CompileKey` / warmup 一致性 | **是** —— 模块级 `if not is_cuda_alike(): pytest.skip(..., allow_module_level=True)`（`:11-12`），**CPU-only 机器上收集到 0 个测试** |
| `test_mhc_tilelang_jit.py` | 119 | TileLang JIT 装饰器惰性化、HIP 符号不被劫持 | 平台门控 |

> ⚠️ **注意上表的"行"列**：其中 5 个测试的宿主是 **`test_mhc_jit_warmup.py`**，
> 不在 `test_mhc_kernels.py` 里（§8.2 已逐行标明）。

**门控条件**：CUDA 单测统一用 `@pytest.mark.skipif(not HAS_TILELANG_MHC, ...)`。
`HAS_TILELANG_MHC = _has_tilelang_mhc()`，在 CUDA 上只要求 `has_tilelang()` 为真
（`layers/mhc.py:16-27`），**不含任何 capability 判断** → **装了 tilelang 的 H200 会真正执行这些 UT**。

### 8.2 逐结构覆盖（sm90 / H200 视角）

#### C11 — 投影 GEMM

| 测试 | 行 | 参数 / 说明 |
|---|---|---|
| `test_hc_prenorm_gemm_tilelang` | 1025 | `num_tokens × hidden_size` = {1,64,512,2048} × {1280,4096,7168}，共 **11 组**；覆盖 `n_splits=1` 与 `num_tokens≥1024` 触发的 `block_m` 变体 |
| `test_hc_prenorm_gemm_dispatch_matches_legacy_runtime_config` | **`test_mhc_jit_warmup.py:64`** | dispatch 结果必须等于运行时 config（**不在 `test_mhc_kernels.py`**） |
| `test_tilelang_warmup_inputs_reproduce_compile_key` | **`test_mhc_jit_warmup.py:300`** | warmup 张量形状能复现 CompileKey（**不在 `test_mhc_kernels.py`**） |

#### C2 — 坍缩 + 门控（含末端 collapse）

| 测试 | 行 | 参数 / 说明 |
|---|---|---|
| `test_mhc_pre_tilelang` | 952 | **核心**：`num_tokens{1,4,8,128}` × `hidden_size{4096,7168}` × `fused_norm{F,T}`，对 `mhc_pre_ref` 比对，**含 Sinkhorn 全路径** |
| `test_deepseek_v41_mhc_pre_delayed` | 227 | **V4.1 延迟版专测**：`(num_tokens,sinkhorn_repeat,norm_eps)` × `entry{broadcast,identity,carried}` × `use_deep_gemm{F,T}` × `fused_norm{F,T}`，并验证 CUDA graph replay 与 eager 逐位一致 |
| `test_mhc_pre_delayed_custom_op_supports_compile` | 752 | `torch.library.opcheck`（自定义算子契约） |
| `test_mhc_pre_big_fuse_dispatch_matches_legacy_runtime_config` | **`test_mhc_jit_warmup.py:81`** | dispatch 一致性（**不在 `test_mhc_kernels.py`**） |
| `test_hc_collapse_preserves_weighted_residual_sum` | 103 | **末端坍缩**：尾块、strided 输入、FP32 混合不变量；含 one-hot 透选与反号相消两个 bitwise 断言 |
| `test_hc_collapse_custom_op_supports_compile` | 141 | 同上契约 |
| `test_hc_collapse_before_gather_is_bitwise_equal` | 194 | TP4/8 下"先坍缩后 gather"逐位一致 |
| `test_v41_dspark_head_collapses_with_last_ffn_mix` | 149 | V4.1 dspark 末端走 `ffn_pre` |
| `test_deepseek_v41_decoder_mixes_match_torch` | 548 | **整层集成**：`entry{broadcast,pipeline,residual,engram}` × `mhc_mode{disabled,eager,overlap,piecewise,large_batch}`；`entry=pipeline` 即 identity 退化 |
| `test_deepseek_v41_capture_previous_aux` | 689 | `capture_aux` 走位（含 Engram 分支） |

#### C3 — post

| 测试 | 行 | 参数 / 说明 |
|---|---|---|
| `test_mhc_post_tilelang` | 1059 | 独立 kernel UT：`num_tokens{1,4,8,128}` × `hidden_size{4096,7168}` × `hc_mult=4` |
| `test_mhc_jit_warmup.py` 相关项 | 165-300 | `MhcPostTileLangKernel` 的 CompileKey / warmup |

参考实现两版齐备：`mhc_post_ref`（`comb.mT` 版，L916）与 `mhc_post_torch`。

#### C5 — 融合 post+pre

| 测试 | 行 | 参数 / 说明 |
|---|---|---|
| `test_deepseek_v41_mhc_fused_post_pre_delayed` | 400 | **V4.1 延迟融合主测**：`carried{F,T}` × `num_tokens{0,1,8,128}` × `hidden_size{4096,7168}`；校验 **6 元组逐项 shape**（L425-435） |
| `test_mhc_fused_post_pre` | 1086 | 非延迟融合（即 **C4**） |
| `test_mhc_fused_post_pre_delayed_custom_op_supports_compile` | 510 | `carried × capture_aux` 二维参数化 |
| `test_mhc_fused_dispatch_matches_the_launch_config` | **`test_mhc_jit_warmup.py:127`** | **决策 1 的形状判据**：(1,4096)→(2,8)、(4,8192)→(2,8)、(8,5120)→(2,8)、(16,5120)→(6,8)、(32,7168)→(6,8)（**参数值正确，但宿主文件不是 `test_mhc_kernels.py`**） |
| `test_mhc_fused_declines_shapes_it_cannot_tile` | **`test_mhc_jit_warmup.py:157`** | **决策 1 的退化分支**：`(33,5120,4) → None`（同上） |
| `test_deepseek_v41_mhc_overlap_preserves_outputs_on_replay` | 311 | ⛔ 硬门控 `is_device_capability_family(100)` → **H200 skip** |

专用参考实现：`mhc_fused_post_pre_delayed_ref`（L375）。

### 8.3 ★ 覆盖缺口：C11 的 DeepGEMM TF32 分支

这是本次核对发现的实质性空白 —— 但**必须按"两条路径"分别看**，不能笼统说"H200 走哪条"：

| | TileLang FP32（`hc_prenorm_gemm_tilelang`） | DeepGEMM TF32（`tf32_hc_prenorm_gemm`） |
|---|---|---|
| 独立 UT | ✅ `test_hc_prenorm_gemm_tilelang`，**11 组** shape | ❌ **没有** `test_tf32_hc_prenorm_gemm` |
| 间接覆盖 | ✅ 另有 `test_deepseek_v41_mhc_pre_delayed` 的 `fused_norm` 等参数化 | ⚠️ 仅靠同一个测试的 `use_deep_gemm=True` 参数化，且限 `hc_mult=4, hidden_size=5120` |
| 前置条件 | `HAS_TILELANG_MHC`（装了 tilelang 即可） | 还需 `is_deep_gemm_supported()`，否则该参数 **skip** |

**对 H200 的含义 —— 覆盖厚薄与执行路径正好错配**：

```
H200：is_deep_gemm_supported() == True（platforms/cuda.py:689-695 含 is_device_capability(90)）
        │
        ├─ T ≤ 32（decode）→ 融合路径 → 恒 TileLang FP32 ──▶ 覆盖【最厚】（11 组独立 UT）
        │
        └─ T > 32（prefill）→ 分离路径 → 命中 DeepGEMM TF32 ──▶ 覆盖【最薄】
                                                               （无独立 UT，仅 1 组间接参数化，
                                                                且需 is_deep_gemm_supported() 才不 skip）
```

也就是说：**H200 上唯一能走到 TF32 的那条路（大批量 prefill），恰好是测试覆盖最薄的一条**；
而覆盖最厚的 TileLang FP32，在 H200 上主要由小批量 decode 走到。
下一轮画 fallback 分支图时，应把这个"错配"标在**分离路径 / TF32 分支**上作为风险提示。

> **注意区分**：不要把这个"错配"简化成"TF32 是 H200 的默认路径" —— 对 decode（T ≤ 32）而言
> H200 走的是 TileLang FP32。原因见 §2.2 / §7.6。
> 补充一点：`tilelang.py` 的**融合**分支（`:386-397`）里根本没有 `is_deep_gemm_supported()` 这个判断，
> 所以那条路上 `use_deep_gemm` 参数化**不会产生任何分支差异** —— 这也解释了为什么该参数化
> 只对分离形状有意义。

### 8.4 其他缺口（均有合理解释）

| 项 | 状态 | 原因 |
|---|---|---|
| C5「融合 vs 分离」的**分离路径** | 无等价 UT | C3 / C11 / C2 已各自单测，"分离"只是编排顺序 |
| C3 的 `write_kv` 实例（layer 36） | 无独立 UT | 属层间集成，由 decoder 级测试覆盖 |
| C3 的 Engram 层实例（layer 1/14） | 无独立 UT | 由 `test_deepseek_v41_decoder_mixes_match_torch` 的 `entry=engram` 覆盖 |

### 8.5 建议的最小验证命令

```bash
# dispatch / CompileKey 一致性（需要 CUDA：模块级 is_cuda_alike() 门控）
python -m pytest tests/kernels/mhc/test_mhc_jit_warmup.py -v

# C2 + C11 的数值正确性（H200 上会执行，含 11 组 prenorm shape）
python -m pytest tests/kernels/mhc/test_mhc_kernels.py \
  -k "prenorm_gemm or mhc_pre_tilelang or mhc_pre_delayed" -v

# C3 + C5
python -m pytest tests/kernels/mhc/test_mhc_kernels.py \
  -k "mhc_post_tilelang or fused_post_pre" -v

# 层间集成（覆盖 C8 的 Engram 特例与三入口）
python -m pytest tests/kernels/mhc/test_mhc_kernels.py \
  -k "decoder_mixes or capture_previous_aux" -v

# 末端坍缩
python -m pytest tests/kernels/mhc/test_mhc_kernels.py -k "hc_collapse" -v
```

---

## 9. ★ 与上一代 DeepSeek-V4 的差异

V4 = `vllm/models/deepseek_v4/`；V4.1 = `vllm/models/deepseek_v41/`。**差异集中在"延迟 seam"与"hc_head 的去留"**。

### 9.1 结构性 diff

| 方面 | DeepSeek-V4 | **DeepSeek-V4.1** | 依据 |
|---|---|---|---|
| **坍缩时机（seam）** | **非延迟**：每个子层自己现算 `pre_mix`，用 `MHCPreOp` / `mhc_pre_tilelang` | **延迟 / shifted**：子层 *k* 的输入坍缩用子层 *k−1* 携带来的 `pre_mix`，同时产出 `next_pre_mix` 给 *k+1*；算子换成 `MHCPreDelayedOp` / `mhc_pre_delayed_tilelang` | `v41/nvidia/ops/mhc.py:305`；V4.1 全树对 `MHCPreOp` / `mhc_pre_tilelang` / `MHCFusedPostPreOp` / `mhc_fused_post_pre_tilelang` **0 命中**（已核实） |
| **post+pre 融合算子** | `MHCFusedPostPreOp` / `mhc_fused_post_pre_tilelang`（非延迟版） | `mhc_fused_post_pre_delayed_tilelang`（延迟版，**sm90 唯一主落点**） | `layers/mhc.py:768-1010` vs `layers/mhc.py:308-559` |
| **learned `hc_head`** | **有**：`hc_head_fn` / `hc_head_base` / `hc_head_scale`，配 `HCHeadOp`（注册名 `"hc_head"`）+ `hc_head_triton` / `hc_head_fused_kernel_tilelang` | **删除**。末端坍缩改由**最后一层 FFN 的 pre-mix** 驱动 `hc_collapse_triton`（`model.py:1070`，随后 `:1071` 接 `self.norm`） | **检查点硬证据**：V4.1 的 `hc_head*` key 数 = **0**（共 96085 key）；`hc_*` = **258** = 43 层 × 6，且**只有 6 种后缀**：`hc_{attn,ffn}_{fn,base,scale}`。源码注释亦自述 *"v4.1 has no learned hc_head"*（`model.py:1066-1068`） |
| **per-sublayer 权重** | 只有 `hc_attn_*`（一套） | **拆成两套**：`hc_attn_*` 与 `hc_ffn_*` | 检查点 6 种后缀；`model.py:409-453` 每层 6 个 fp32 参数 |
| **dspark / MTP 的 head** | `v4/nvidia/dspark.py` 有 `hc_head_fn/_base/_scale`（`:160-167, 274-278`） | **无**（参数路由表也删掉了这三个条目）；dspark 构造 `DeepseekV4DecoderLayer` 时**不传** `mhc_stream` / `fuse_mhc_all_reduce`（`dspark.py:126-133`）→ **草稿层恒走非 overlap 路径** | `v41/nvidia/dspark.py:141-156, 242, 343-349` |
| **entry broadcast 的 `fn`** | 走 `mhc_pre_broadcast_tilelang`（专用 kernel，且强制 `use_tilelang_fallback=False`） | **Python 侧完成**：`residual = x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()` 后调 C2；`fn` 用**运行时派生**的 `hc_attn_fn_broadcast`（= `hc_attn_fn.detach().view(-1, hc, H).sum(dim=1)`，在 `finalize_mhc_broadcast_weights()` 里生成，`model.py:1418-1422`），**检查点里没有这个 key** | `model.py:500-514, 1413-1426` |
| **`hc_head` 算子族现状** | 在用 | **仍注册、仍被 warmup 编译，但对 V4.1 是死路径**：`HCHeadOp`（`layers/mhc.py:648-767`）、`hc_head_fused_kernel_tilelang`（`tilelang.py:1041-1125`）、`_HC_HEAD_FUSED_TILELANG_KERNEL`（`tilelang_kernels.py:1873`）都还在 | `test_mhc_jit_warmup.py` 的"5 个 kernel"计数里就包含它 |
| **Engram** | — | **V4.1 特例**：`elif self.engram is not None and engram_hashes is not None` 出现在 `else: mhc_shifted_post_pre(...)` **之前**，且 Engram 注入落在"上一子层 post"与"本块 pre"**之间** → post **无法**被折进 pre-norm GEMM 的 prologue，故必须拆成 post + engram + pre 三步 | `model.py:531-561`；`engram_layer_ids=[1,14]`（`config.json:131-134`） |

### 9.2 ★ 同仓库其他"HC / iHC / HyperConnection"（强制鉴别，勿混淆）

全树有 **4 个模型族各自定义** `hc` / `mhc`，语义**互不相同**：

| 实现 | 文件 | 是什么 | 与 DSV4.1 mHC 的关系 |
|---|---|---|---|
| **DSV4.1 mHC** | `models/deepseek_v41/nvidia/ops/{mhc,mega_mhc}.py` + `model_executor/layers/mhc.py` + `kernels/mhc/**` | 多流 hyper-connection：**comb 矩阵 + Sinkhorn 归一化 + learned `fn` 投影** | **本对象** |
| **DSV4.0 mHC** | `models/deepseek_v4/**` | 上一代：**非延迟 seam + learned `hc_head`** | ⚠️ **同名不同代**，seam 与 head 都不同（见 §9.1） |
| **HY V4 iHC** | `models/hy_v4/nvidia/hc.py`、`models/hy_v4/nvidia/triton_ihc.py`、`model_executor/layers/hpc/hpc_ihc.py` | **iHC = independent Hyper-Connections**：`hc_mult` 条并行残差通道，但 **pre/post/head 三步只有 sigmoid 门 + 逐通道加权，没有 comb 矩阵、没有 Sinkhorn**；另有 HPC 单 kernel 替换（`HpcIHCPre/Post/Head`） | ❌ **不同的数学**。名字像、结构不同 |
| **Qwen4Exp HyperConnection** | `models/qwen4_exp/nvidia/ops/hc.py`、`models/qwen4_exp/amd/ops/hc.py`、`.../cute_dsl/hc_down_silu.py` | Qwen4Exp 自己的 HyperConnection kernel（含 `_grouped_gemma_rmsnorm_kernel` 等） | ❌ 另一族，无共享符号 |

> **鉴别要点（三问）**：① 有没有 **comb 矩阵 + Sinkhorn**？有 → DSV4.x mHC；没有 → iHC。
> ② 有没有 **learned `hc_head`**？有 → V4.0；没有 → V4.1。
> ③ seam 是 **延迟（携带 `pre_mix`）** 还是现算？延迟 → V4.1；现算 → V4.0。

---

## 10. `torch.ops.vllm.*` 注册表全量（9 个 op）

`direct_register_custom_op` 在 mHC 相关文件中共注册 **9 个** op：

| op_name | 文件:行 | V4.1 是否可达 |
|---|---|---|
| `mhc_pre_delayed_tilelang` | `tilelang.py:1089` | ✅ **主 seam（C2）** |
| `mhc_fused_post_pre_delayed_tilelang` | `tilelang.py:1095` | ✅ **主落点（C5）** |
| `mhc_pre_tilelang` | `tilelang.py:1101` | ❌ V4.0 专用（V4.1 全树 0 命中） |
| `mhc_post_tilelang` | `tilelang.py:1107` | ✅ C3（Engram / 收尾 / dspark） |
| `mhc_fused_post_pre_tilelang` | `tilelang.py:1114` | ❌ V4.0 专用（非延迟融合） |
| `hc_head_fused_kernel_tilelang` | `tilelang.py:1121` | ❌ **死路径**：仍注册、仍 warmup，但 V4.1 无 `hc_head`（检查点 0 key） |
| `hc_head_triton` | `triton.py:330` | ❌ 同上（V4.0 head 的 fallback） |
| `hc_collapse_triton` | `triton.py:337` | ✅ **末端坍缩**（由最后一层 FFN pre-mix 驱动） |
| `mhc_pre_mix_triton` | `triton.py:345` | ❌ AITER / ROCm 专用 |

> ⚠️ 注意 `hc_head_triton` 的定位：`triton.py:330` 注册它，`layers/mhc.py:714` 只在 **`forward_hip`（ROCm）** 分支调用；
> CUDA 的 `forward_cuda`（`:662-682`）调的是 **`torch.ops.vllm.hc_head_fused_kernel_tilelang`**（`:674`），
> **不是** `hc_head_triton`。两者别混。
