# DeepSeek-V4.1-Flash 在 vLLM 中运行涉及的算子清单（NVIDIA）

- 模型：`/home/xuqiankun/Data/DeepSeek-V4.1-Flash`（`DeepseekV41ForCausalLM`，FP8 block-wise 量化 `[32,32]` + FP4 experts，含 DSA 稀疏注意力 / KV compressor / MHC hyper-connection / engram / dspark-dflash MTP）
- vLLM 仓库：`/home/xuqiankun/Code/vllm_latest`
- trace 来源：
  - `/home/xuqiankun/Code/profiler.log`（汇总）
  - `/home/xuqiankun/Code/dp0_pp0_tp0_dcp0_ep0_rank0.1790403446596720255.pt.trace.json`（**真实 `llm.start_profile()/llm.chat()/llm.stop_profile()` 采集**，8×NVIDIA H200，sm90；约 262 万个事件，其中 GPU kernel 10221 次启动、cpu_op 72696 次）
- 模型实现主体：`vllm/models/deepseek_v41/`（`nvidia/model.py` 为入口，复用 `vllm/models/deepseek_v4/` 的部分公共算子）

下表按功能分类，**仅覆盖 NVIDIA（CUDA）路径**。"trace 次数"列为真实 trace 中 cpu_op 调用次数（triton/tilelang kernel 直接启动的为 kernel 启动次数）。unit test 均为 `tests/` 下的相对路径。

## 1. 注意力（MLA 稀疏注意力 / Indexer / Compressor）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `_flashmla_C::sparse_decode_fwd`（kernel: `sm90::decode::sparse::flash_fwd_splitkv_mla_fp8_sparse_kernel`、`smxx::decode::flash_fwd_mla_combine_kernel`、`smxx::decode::get_mla_metadata_kernel`） | 175 | FlashMLA 稀疏注意力 decode | 封装 `vllm/v1/attention/backends/mla/flashmla_sparse.py`；模型侧调用 `vllm/models/deepseek_v41/nvidia/flashmla.py:351`；构建 `cmake/external_projects/flashmla.cmake` | `tests/kernels/attention/test_flashmla_sparse.py`、`tests/kernels/attention/test_flashmla.py` |
| `_flashmla_C::sparse_prefill_fwd`（kernel: `sm90::prefill::sparse_fwd::sparse_attn_fwd_kernel`） | 40 | FlashMLA 稀疏注意力 prefill | 同上 | 同上 |
| `_flashmla_C::fused_norm_rope_attn_rope_cast_decode` / `..._fwd`、`permute_q_b_proj` | 本 trace 未命中 | DSV4.1 mega-attention 融合算子（norm+rope+attn 一体） | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py:361,465` | `tests/kernels/test_dsv41_mega_attn_layouts.py`；benchmark: `benchmarks/kernels/benchmark_dsv41_mega_attn.py` |
| `_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`（kernel: `vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel`，含 `ReducedGrid` 变体 ×40） | 230 | Q norm + RoPE + KV RoPE + FP8 量化 + 写 KV cache 融合算子 | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1134`；注册 `csrc/libtorch_stable/torch_bindings.cpp:426,808`；调用 `vllm/models/deepseek_v41/attention.py:1044` | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`、`tests/kernels/test_compressor_kv_cache.py` |
| `_C::fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` / `_fp8_insert` | 本 trace 未命中 | 上述融合算子的 full-cache 变体 | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1228` 起；注册 `torch_bindings.cpp:436,441,811,814`；调用 `vllm/models/deepseek_v41/attention.py:1072,1087` | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| `_C::fused_deepseek_v4_kv_rope_insert` | 本 trace 未命中 | dspark draft 层 KV RoPE + 写 cache | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1049`；注册 `torch_bindings.cpp:420,806`；调用 `vllm/models/deepseek_v41/nvidia/dspark.py:268` | `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| `vllm::sparse_attn_indexer` | 40 | DSA lightning indexer（稀疏注意力 token 选择） | `vllm/model_executor/layers/sparse_attn_indexer.py`、`vllm/v1/attention/backends/mla/indexer.py` | `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`、`tests/model_executor/layers/test_mla_short_prefill_indexer.py`、`tests/v1/attention/test_indexer_dcp_localize.py`、`tests/v1/attention/test_indexer_native_next_n.py` |
| `deep_gemm::sm90_fp8_mqa_logits`（prefill） | 8 | indexer 的 FP8 MQA logits（DeepGEMM） | `vllm/v1/attention/backends/mla/indexer.py`（经 `vllm/utils/deep_gemm.py` 封装） | `tests/kernels/attention/test_deepgemm_attention.py` |
| `deep_gemm::sm90_fp8_paged_mqa_logits` + `deep_gemm::sched::sm90_paged_mqa_logits_metadata`（decode，paged KV） | 20+12 / 8 | indexer decode 路径的 paged MQA logits | 同上 | `tests/kernels/attention/test_deepgemm_attention.py` |
| `_C::fused_indexer_q_rope_quant` | 本 trace 未命中 | indexer Q 的 RoPE+量化融合 | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | `tests/kernels/test_fused_indexer_q_rope_quant.py` |
| `_indexer_k_norm_rope_quant_store_kernel`（triton） | 20 | indexer K 的 norm+RoPE+量化+写 cache | `vllm/models/deepseek_v41/common/ops/indexer_k_store.py:123` | `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` |
| `_C::top_k_per_row_prefill`（kernel: `vllm::topKPerRowPrefill<512>`） | 8 | DSA prefill 阶段每行 top-k 选择 | 注册 `csrc/libtorch_stable/torch_bindings.cpp:858` | `tests/kernels/test_top_k_per_row.py` |
| `_C::cooperative_topk`（kernel: `vllm::cooperative::cooperative_topk_cs8<512u>`） | 32 | DSA top-k 的另一后端（cooperative） | 注册 `csrc/libtorch_stable/torch_bindings.cpp:862` | `tests/kernels/test_top_k_per_row.py`（`backend="cooperative_topk"`） |
| DSA candidate blocks 组：`_block_scores_kernel`、`_candidate_flags_kernel`、`_mask_candidates_kernel`、`_store_candidates_kernel`（triton） | 5 / 20 / 20 / 5 | DSA candidate block 打分/掩码/存储 | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:15,78,109,55` | 未找到直接单测 |
| `_C_cache_ops::cp_gather_indexer_k_quant_cache`（kernel: `vllm::cp_gather_indexer_k_quant_cache_kernel<32>`） | 8 | CP（context parallel）下 gather indexer K 量化 cache | 注册 `csrc/libtorch_stable/torch_bindings.cpp:1167` | `tests/v1/attention/test_indexer_dcp_localize.py`（间接） |
| `_compute_swa_indices_and_lens_kernel`（triton） | 40 | 滑动窗口（SWA）层的稀疏索引/长度计算 | `vllm/v1/attention/backends/mla/sparse_swa.py:1008` | `tests/v1/attention/test_deepseek_v4_swa_visible.py` |
| `_compute_global_topk_indices_and_lens_kernel`（triton） | 152 | compressor 全局 topk block 索引/长度计算 | `vllm/models/deepseek_v41/common/ops/cache_utils.py:753` | `tests/kernels/test_compressor_kv_cache.py`（部分覆盖） |
| compressor 组：`_fused_save_compress_norm_kernel`、`_rope_quant_insert_kernel`（triton） | 20 / 20 | KV compressor：压缩统计 norm 保存、RoPE+量化+写压缩 cache | `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py:118,304`、`vllm/models/deepseek_v41/compressor.py` | `tests/kernels/test_compressor_kv_cache.py` |
| `_ring_slot_mapping_kernel`（triton） | 5 | compressor ring buffer slot 映射 | `vllm/models/deepseek_v41/compressor.py:61` | `tests/kernels/test_compressor_kv_cache.py` |
| `_gather_lookback_kernel`（triton） | 5 | V4.1 lookback/replay 隐状态 gather | `vllm/models/deepseek_v41/nvidia/model_state.py:32` | `tests/models/test_deepseek_v41_replay_batch.py`、`test_deepseek_v41_replay_start.py`（间接） |
| `vllm::fused_inv_rope_fp8_quant_kernel` | 215 | o_proj 输入的逆 RoPE + FP8 量化（WoA 分解路径） | 定义 `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:352`（注册同文件 :532）；调用 `vllm/models/deepseek_v4/nvidia/ops/o_proj.py:66`、`vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` | `tests/kernels/test_fused_inv_rope_fp8_quant.py`、`tests/quantization/test_fp8.py` |
| `_vllm_fa3_C::fwd` + `vllm::flash_attn_maxseqlen_wrapper`（kernel: `cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90...>>`、`flash::prepare_varlen_num_blocks_kernel`） | 32 | FlashAttention-3（全注意力/SWA 层） | 接口 `vllm/vllm_flash_attn/flash_attn_interface.py:379`；构建 `cmake/external_projects/vllm_flash_attn.cmake` | `tests/kernels/attention/test_flash_attn.py` |
| FlashInfer sparse MLA（备选注意力后端） | 本 trace 未命中 | FlashInfer 实现的稀疏 MLA 后端 | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | `tests/v1/attention/test_flashinfer_sparse_mla_workspace.py`、`tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`、`tests/v1/attention/test_sparse_mla_backends.py`、`tests/v1/attention/test_dspark_noncausal_sparse_mla.py` |

## 2. MoE（路由 / 专家 GEMM / 激活）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `vllm::moe_forward_shared` | 215 | MoE forward 主入口 custom op（含 shared expert 融合） | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:154`（注册 :210） | `tests/model_executor/layers/test_fused_shared_expert.py`、`tests/compile/test_config.py` |
| `_dsv4_topk_kernel`（triton） | 200（kernel 启动） | DSV4 topk 路由（sqrtsoftplus 打分 + noaux_tc），triton 版 | `vllm/model_executor/layers/fused_moe/router/dsv4_topk.py:41` | `tests/kernels/moe/test_topk_softplus_sqrt.py` |
| `_moe_C::topk_softplus_sqrt`（kernel: `vllm::moe::topkGatingSoftplusSqrt<...>`） | 15 | 同上路时的 CUDA 版 | 注册 `csrc/libtorch_stable/moe/torch_bindings.cpp:143` | `tests/kernels/moe/test_topk_softplus_sqrt.py` |
| `_moe_C::moe_align_block_size`（kernel: `vllm::moe::moe_align_block_size_kernel`、`count_and_sort_expert_tokens_kernel`） | 215 | MoE 专家 token 对齐/排序 | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`；注册 `csrc/libtorch_stable/moe/torch_bindings.cpp:145` | `tests/kernels/moe/test_moe_align_block_size.py` |
| `_moe_C::moe_sum`（kernel: `vllm::moe::moe_sum_vec_kernel` ×200 + `moe_sum_vec_dynamic_kernel` ×15） | 215 | topk 专家输出加权求和 | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:948`；注册 `csrc/libtorch_stable/moe/torch_bindings.cpp:144` | `tests/kernels/moe/test_moe.py` |
| `_C::marlin_gemm`（kernel: `marlin::Marlin<...>` 多种模板实例） | 1145 | Marlin 量化 GEMM（W4A16 路径的线性层/专家，trace 中最高频的自定义 GEMM） | `csrc/libtorch_stable/quantization/marlin/marlin.cu:734` | `tests/kernels/quantization/test_marlin_gemm.py`、`tests/kernels/quantization/test_marlin_tile_padding.py` |
| `_moe_C::moe_wna16_marlin_gemm`（kernel: `marlin_moe_wna16::Marlin<...>`） | 430 | Marlin MoE WNA16 专家 GEMM | `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu:705` | `tests/quantization/test_moe_wna16.py`、`tests/kernels/moe/test_moe.py`、`tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py` |
| `_C::silu_and_mul`（kernel: `vllm::act_and_mul_kernel`，clamp 关闭实例） | 32 | SwiGLU 激活（shared expert / dense MLP） | 注册 `csrc/libtorch_stable/torch_bindings.cpp:870` | `tests/kernels/core/test_activation.py` |
| `_C::silu_and_mul_with_clamp`（kernel: `vllm::act_and_mul_kernel`，clamp 开启实例） | 215 | 带 swiglu_limit 截断的 SwiGLU（模型 `swiglu_limit=10.0`） | 注册 `csrc/libtorch_stable/torch_bindings.cpp:884` | `tests/kernels/core/test_activation.py` |
| `_swiglu_limit_pad_aware_kernel`（triton） | 215 | MoE 分组 GEMM 后 padding 感知的 swiglu_limit 激活 | `vllm/model_executor/layers/fused_moe/utils.py:492,547` | `tests/kernels/moe/test_batched_moe.py`、`tests/models/test_deepseek_v4_mega_moe.py` |
| `deep_gemm::...`（block FP8 分组 GEMM 等） | 本 trace 未命中 | FP8 block-wise MoE/dense GEMM（DeepGEMM，备选路径） | 封装 `vllm/utils/deep_gemm.py` | `tests/kernels/moe/test_deepgemm.py`、`tests/kernels/moe/test_batched_deepgemm.py`、`tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py`、`tests/model_executor/test_deep_gemm_warmup.py` |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinea...`（如 `LLBf16SplitK` ×175、cuTe DSL scaled_mm ×40/40/20/18） | ~293 | CUTLASS/cuTe DSL FP8 scaled_mm（block FP8 线性层） | `vllm/model_executor/layers/quantization/utils/`（cutlass fp8） | `tests/kernels/quantization/test_cutlass_scaled_mm.py`、`tests/kernels/quantization/test_block_fp8.py`、`tests/kernels/quantization/test_scaled_mm_kernel_selection.py` |

## 3. MHC（Hyper-Connection，tilelang JIT / triton / DeepGEMM）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 430 | MHC 前置融合（含 norm） | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:299` | `tests/kernels/mhc/test_mhc_kernels.py`、`tests/kernels/mhc/test_mhc_tilelang_jit.py` |
| `mhc_fused_tilelang_kernel` | 333 | MHC 融合混合 | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:668` | 同上 |
| `mhc_post_tilelang_kernel` | 97 | MHC 后置（残差回写） | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:789`；注册 `torch.ops.vllm.mhc_post_tilelang`（`vllm/model_executor/kernels/mhc/tilelang.py:791,1107`）；调用 `vllm/models/deepseek_v41/nvidia/model.py:1095` | 同上 |
| `vllm::hc_collapse_triton`（triton） | —（直接启动） | MHC 隐状态折叠（pre_mix 混合回主干） | `vllm/model_executor/kernels/mhc/triton.py`；调用 `vllm/models/deepseek_v41/nvidia/model.py:1013` | `tests/kernels/mhc/test_mhc_kernels.py` |
| `_hc_head_reduce_store_kernel`（triton） | 10 | MHC head 规约回写 | `vllm/model_executor/kernels/mhc/triton.py:66` | `tests/kernels/mhc/test_mhc_kernels.py` |
| `deep_gemm::sm90_tf32_hc_prenorm_gemm_impl`（多个模板实例：20480×79、5120×4+1、另一 20480 实例 ×13） | 97 | HC prenorm GEMM（DeepGEMM tf32） | 封装 `vllm/utils/deep_gemm.py:931`；调用 `vllm/models/deepseek_v41/nvidia/ops/mhc.py:182`；tilelang 备选 `vllm/model_executor/kernels/mhc/tilelang_kernels.py:842` | `tests/kernels/mhc/test_mhc_kernels.py`（tilelang 路径）；DeepGEMM 路径无独立单测 |

## 4. Engram（n-gram 记忆嵌入）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `_engram_lookup_kernel`（triton） | 10 | engram embedding 查表 | `vllm/models/deepseek_v41/common/engram.py:628` | `tests/kernels/test_engram.py`、`tests/distributed/test_engram_dp_shard.py` |
| `_hash_ids_kernel`（triton） | 5 | engram n-gram hash id 计算 | `vllm/models/deepseek_v41/common/engram.py:277` | 同上 |
| `_fused_engram_post_wkv_kernel`（triton） | 10 | engram post-wkv 融合 | `vllm/models/deepseek_v41/common/engram.py:1156`（启动 :1508） | 同上 |

## 5. 采样与投机解码（gumbel / rejection sampler / dspark / dflash MTP）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `_gumbel_sample_kernel`（triton） | 26 | gumbel-max 采样 | `vllm/v1/worker/gpu/sample/gumbel.py:274` | `tests/v1/sample/test_sampler.py`、`tests/watermarking/test_gumbel.py` |
| rejection sampler 组：`_compute_local_logits_stats_kernel`、`_rejection_kernel`、`_resample_kernel`、`_insert_resampled_kernel`（triton） | 各 4 | 投机解码拒绝采样 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:194,485,731,946` | `tests/v1/spec_decode/test_rejection_sampler_utils.py`、`tests/v1/sample/test_rejection_sampler.py` |
| `_prepare_dflash_inputs_kernel`（triton） | 15 | dflash 投机解码输入准备 | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:522`（启动 :766） | `tests/v1/spec_decode/test_dflash_prepare_inputs.py`、`tests/v1/spec_decode/test_dflash2.py`、`tests/v1/spec_decode/test_dflash_lookahead.py`、`tests/v1/spec_decode/test_dflash_causality.py` |
| `_prepare_prefill_inputs_kernel`（draft_model，triton） | 1 | draft 模型 prefill 输入准备 | `vllm/v1/worker/gpu/spec_decode/draft_model/speculator.py:400` | `tests/v1/spec_decode/`（dflash/dspark 系列） |
| input_batch 组：`_prepare_pos_seq_lens_kernel`、`_combine_sampled_and_draft_tokens_kernel`、`_get_num_sampled_and_rejected_kernel`、`_post_update_kernel`、`_expand_idx_mapping_kernel`（triton） | 各 5 | spec decode 后 input batch 更新 | `vllm/v1/worker/gpu/input_batch.py:347,405,507,556,690` | `tests/v1/worker/`、`tests/v1/spec_decode/`（间接） |
| dspark MTP 相关（复用 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`、`mhc_post_tilelang` 等） | — | dspark draft 层（`dspark_target_layer_ids`） | `vllm/models/deepseek_v41/nvidia/dspark.py`、`vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | `tests/models/test_dspark_mla.py`、`tests/v1/worker/test_dspark_utils.py`、`tests/v1/spec_decode/test_dsv4_mtp_local_argmax.py` |

## 6. v1 worker / 调度辅助 kernel（triton）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `_token_request_mapping_kernel` | 65 | attention metadata：token→request 映射 | `vllm/v1/attention/ops/metadata.py:22` | 经各 attention backend 测试间接覆盖 |
| `_zero_kv_blocks_kernel` | 2 | KV block 清零 | `vllm/v1/worker/utils.py:60` | `tests/v1/worker/test_kv_block_zeroer.py` |
| `_gather_block_tables_kernel`、`_compute_slot_mappings_kernel` | 各 5 | block table gather / slot mapping 计算 | `vllm/v1/worker/gpu/block_table.py:247,287` | `tests/v1/worker/`（间接） |
| `_apply_write_kernel` | 5 | 异步写回缓冲区应用 | `vllm/v1/worker/gpu/buffer_utils.py:301` | `tests/v1/worker/`（间接） |

## 7. 通用层 custom op（embedding / norm）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `_C::vocab_parallel_embedding`（kernel: `vllm::vocab_embedding::vocab_parallel_embedding_kernel`） | 10 | TP 词表并行 embedding | 注册 `csrc/libtorch_stable/torch_bindings.cpp:746`；层 `vllm/model_executor/layers/vocab_parallel_embedding.py:204` | `tests/kernels/core/test_vocab_parallel_embedding.py` |
| `_C::rms_norm`（kernel: `vllm::rms_norm_kernel`） | 30 | 未被融合的 RMSNorm（融合之外的少数 norm 点） | `vllm/model_executor/layers/layernorm.py` | `tests/kernels/core/test_layernorm.py` |
| `_C::get_cuda_view_from_cpu_tensor` | 1 | CPU pinned tensor 的 CUDA view（工具算子） | 注册 `csrc/libtorch_stable/torch_bindings.cpp:905` | — |

## 8. 通信（TP/DP）

| profiler 算子 | trace 次数 | 说明 | 源码位置 | Unit test |
|---|---|---|---|---|
| `vllm::all_reduce`（kernel: `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` ×359、`multimem_all_reduce_kernel` ×81） | 440 | 自定义 all-reduce（MNNVL/flashinfer oneshot、NCCL symmetric-memory multimem 路径） | `vllm/distributed/device_communicators/`（custom_all_reduce、symm_mem、MNNVL 相关） | `tests/distributed/test_custom_all_reduce.py`、`tests/distributed/test_symm_mem_allreduce.py`、`tests/distributed/test_nccl_symm_mem.py`、`tests/distributed/test_flashinfer_pcie_ipc_all_reduce.py`、`tests/distributed/test_quick_all_reduce.py` |
| `symm_mem::multimem_all_reduce_` | 81 | NCCL symmetric memory multimem all-reduce 的调度 op | `vllm/distributed/`（symm_mem） | `tests/distributed/test_nccl_symm_mem.py`、`tests/distributed/test_symm_mem_allreduce.py` |
| `vllm::all_gather`（kernel: `ncclDevKernel_AllGather_RING_LL`） | 21 | NCCL all-gather（DP 场景） | `vllm/distributed/` | `tests/distributed/test_comm_ops.py`、`tests/distributed/test_custom_all_gather_reduce_scatter.py` |

## 9. 通用 aten / cuBLAS（非 vLLM 自定义，无仓库内单测对象）

真实 trace 中 aten cpu_op 共 72696 次，Top：`aten::empty` ×17612、`aten::as_strided` ×9752、`aten::view` ×8600、`aten::slice` ×5792、`aten::reshape` ×3192、`aten::copy_` ×929、`aten::linear` ×226、`aten::mm` ×219、`aten::bmm` ×215、`aten::cat` ×111、`aten::addmm` ×67、`aten::argmax` ×26、`aten::embedding` ×25、`aten::topk` ×5。底层 GPU kernel：

- `nvjet_sm90_*`（cuBLASLt 启发式 GEMM，含 `splitKreduce_kernel` ×215 等）
- `at::native::elementwise_kernel` / `vectorized_elementwise_kernel` / `reduce_kernel`（add/mul/fill/copy/mean/rsqrt/pow 等）
- `at::native::mbtopk::*` / `radixSortKVInPlace`（aten::topk 底层）
- `cutlass_80_tensorop_bf16_s16816gemm` ×2（少量 fallback GEMM）
- `Memcpy DtoD/HtoD` ×318、`gpu_memset` ×98、cuda_runtime 事件 ×22321

## 附：模型级 / 组件级测试（非单算子）

- `tests/models/test_deepseek_v41_decoder_replay_layers.py`、`test_deepseek_v41_replay_batch.py`、`test_deepseek_v41_replay_start.py`
- `tests/models/test_deepseek_v4_fi_moe_ep.py`、`test_deepseek_v4_mega_moe.py`
- `tests/models/multimodal/processing/test_deepseek_v4_vl.py`（视觉部分）
- `tests/tokenizers_/test_deepseek_v41.py`、`tests/parser/engine/test_deepseek_v41.py`、`tests/tool_parsers/test_deepseekv4_tool_parser.py`
- `tests/config/test_model_arch_config.py`、`tests/v1/attention/test_sparse_mla_mask.py`

## 备注

- `_C` / `_moe_C` / `_C_cache_ops` 为 vLLM C++ 扩展的 torch library 命名空间；本仓库使用 libtorch_stable ABI（`csrc/libtorch_stable/`）。
- `deep_gemm::*`、FlashMLA（`_flashmla_C`）、FlashAttention（`_vllm_fa3_C`）的 CUDA kernel 本体在外部项目（cmake external projects），仓库内是封装与调用层。
- `mhc_*_tilelang_kernel` 由 tilelang JIT 在运行时生成，定义在 `vllm/model_executor/kernels/mhc/tilelang_kernels.py`；trace 中还有 513 次匿名 `Kernel` 启动，来自 tilelang/triton JIT 的未命名 kernel。
- RMSNorm 的两种形态：大多数被融合进 MHC 前置 kernel（`mhc_pre_big_fuse_with_norm`）与 `fused_deepseek_v4_qnorm_rope_*` 融合算子；未融合的少数点走独立的 `_C::rms_norm`（trace 中 30 次）。
- 真实 trace 中**未命中**、但代码路径存在的备选/变体：mega-attn 融合算子、full-cache insert 变体、`fused_indexer_q_rope_quant`、DeepGEMM MoE、FlashInfer sparse MLA 后端——是否启用取决于 backend/配置选择。
- trace 中 `gpu_user_annotation` 显示两个执行上下文：`execute_context_0(0)_generation_1(6)` 与 `execute_context_1(2026)_generation_0(0)`。
