import math    
import torch  
import torch.nn.functional as F 
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.scalar_type import scalar_types
from dataclasses import dataclass
from safetensors.torch import load_file  
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config  
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
try:
    from .quant_utils import native_w8a8_block_matmul, native_per_token_group_quant_fp8
except ImportError:    
    from quant_utils import native_w8a8_block_matmul, native_per_token_group_quant_fp8
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.v1.worker.workspace import init_workspace_manager

try:
    from .moe_utils import (
    make_dummy_moe_config,
    modular_triton_fused_moe,
)  # 正常包导入场景
except ImportError:
    from moe_utils import (
    make_dummy_moe_config,
    modular_triton_fused_moe,
)    # 直接运行场景

# ── Step 1: MXFP4 反量化 → BF16 ──────────────────────────────────────────────  
def dequantize_mxfp4(  
    weight_i8: torch.Tensor,    # [N, K//2] uint8，nibble packing  
    scale_e8m0: torch.Tensor,   # [N, K//32] uint8，E8M0 block scale  
) -> torch.Tensor:  
    """MXFP4 → BF16，group_size=32"""  
    E2M1_values = torch.tensor(  
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=weight_i8.device  
    )  
    w_u8 = weight_i8.view(torch.uint8)  
    N, K_half = w_u8.shape  
    K = K_half * 2  
  
    # 解包 nibble：低 4 bit = 偶数索引，高 4 bit = 奇数索引  
    nibbles = torch.zeros(N, K, dtype=torch.uint8, device=weight_i8.device)  
    nibbles[:, 0::2] = w_u8 & 0x0F  
    nibbles[:, 1::2] = (w_u8 >> 4) & 0x0F  
  
    # 解码 E2M1：高位是符号位，低 3 位是幅度码  
    sign = 1 - 2 * ((nibbles & 0b1000) >> 3).float()  
    magnitude = (nibbles & 0b0111).long()  
    x_float = E2M1_values[magnitude.reshape(-1)].reshape(N, K) * sign  
  
    # 应用 E8M0 scale：scale = 2^(e8m0 - 127)，每 32 个元素共享一个 scale  
    scale_f32 = torch.exp2(scale_e8m0.float() - 127.0)  # [N, K//32]  
    x_dequant = x_float.reshape(N, K // 32, 32) * scale_f32.unsqueeze(-1)  
    return x_dequant.reshape(N, K).to(torch.bfloat16)  
  
  
# ── Step 2: BF16 → FP8 block quantization ────────────────────────────────────  
def quantize_to_fp8_block(  
    weight_bf16: torch.Tensor,          # [N, K] bfloat16  
    block_size: list[int] = [128, 128],  
) -> tuple[torch.Tensor, torch.Tensor]:  
    """BF16 → FP8 E4M3，返回 (weight_fp8 [N,K], scale_f32 [N//bn, K//bk])"""  
    N, K = weight_bf16.shape  
    block_n, block_k = block_size  
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0  
  
    n_tiles = math.ceil(N / block_n)  
    k_tiles = math.ceil(K / block_k)  
  
    # Pad to tile boundary  
    pad_n = (block_n - N % block_n) % block_n  
    pad_k = (block_k - K % block_k) % block_k  
    w = F.pad(weight_bf16, (0, pad_k, 0, pad_n)).float()  
  
    # [n_tiles, k_tiles, block_n, block_k]  
    w_blocked = w.view(n_tiles, block_n, k_tiles, block_k).permute(0, 2, 1, 3)  
    amax = w_blocked.abs().amax(dim=(-2, -1)).clamp(min=1e-12)  # [n_tiles, k_tiles]  
    scale = amax / fp8_max  # F32 scale  
  
    w_fp8 = (w_blocked / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max)  
    w_fp8 = w_fp8.permute(0, 2, 1, 3).reshape(n_tiles * block_n, k_tiles * block_k)  
    w_fp8 = w_fp8[:N, :K].to(torch.float8_e4m3fn)  
  
    return w_fp8, scale  # scale shape: [N//128, K//128]  

# ── Step 3: FP8 反量化 → BF16 ──────────────────────────────────────────────  
def per_block_cast_from_fp8(  
    x_fp8: torch.Tensor,   # [M, N] float8_e4m3fn  
    sf: torch.Tensor,      # [M // block_m, N // block_n] float32  
    block_size: list[int] = [128, 128],  
    out_dtype: torch.dtype = torch.bfloat16,  
) -> torch.Tensor:  
    """  
    per_block_cast_to_fp8 的反向操作（dequantize）。  
  
    x_fp8 * sf（per-block 广播）→ out_dtype  
    """  
    M, N = x_fp8.shape  
    block_m, block_n = block_size  
  
    def _align(x, y):  
        return math.ceil(x / y) * y  
  
    M_pad = _align(M, block_m)  
    N_pad = _align(N, block_n)  
  
    # Step 1: pad 到 block boundary（和正向保持一致）  
    x_padded = torch.zeros(M_pad, N_pad, dtype=torch.float32, device=x_fp8.device)  
    x_padded[:M, :N] = x_fp8.float()  
  
    # Step 2: view 成 [M//bm, bm, N//bn, bn]  
    x_view = x_padded.view(M_pad // block_m, block_m, N_pad // block_n, block_n)  
  
    # Step 3: sf [M//bm, N//bn] → [M//bm, 1, N//bn, 1] 广播  
    sf_view = sf.view(M_pad // block_m, 1, N_pad // block_n, 1)  
  
    # Step 4: 反量化  
    x_dequant = x_view * sf_view 
  
    # Step 5: reshape 回 [M_pad, N_pad]，截取 [:M, :N]  
    return x_dequant.view(M_pad, N_pad)[:M, :N].contiguous().to(out_dtype)
  
# ── 完整转换（单 expert）─────────────────────────────────────────────────────  
def convert_mxfp4_to_fp8_block(weight_i8, scale_e8m0, block_size=[128, 128]):  
    bf16 = dequantize_mxfp4(weight_i8, scale_e8m0)  
    # w0, s0 = quantize_to_fp8_block(bf16, block_size)  
    w1, s1 = per_block_cast_to_fp8(bf16, block_size, use_ue8m0=True)
    # w2, s2 = per_block_cast_to_fp8(bf16, block_size, use_ue8m0=False)
    return w1, s1, bf16
  
  
# ── 批量转换（所有 experts）──────────────────────────────────────────────────  
def convert_moe_mxfp4_to_fp8_block(  
    w_i8: torch.Tensor,         # [E, N, K//2] I8  
    w_scale_e8m0: torch.Tensor, # [E, N, K//32] uint8  
    block_size: list[int] = [128, 128],  
) -> tuple[torch.Tensor, torch.Tensor]:  
    E, N, K_half = w_i8.shape  
    K = K_half * 2  
    bn, bk = block_size  
    w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w_i8.device)  
    w_fp16 = torch.empty(E, N, K, dtype=torch.bfloat16, device=w_i8.device)  
    w_scale = torch.empty(  
        E, math.ceil(N / bn), math.ceil(K / bk),  
        dtype=torch.float32, device=w_i8.device  
    )  
    for i in range(E):  
        w_fp8[i], w_scale[i], w_fp16[i] = convert_mxfp4_to_fp8_block(  
            w_i8[i], w_scale_e8m0[i], block_size  
        )  
    return w_fp8, w_scale, w_fp16

def load_all_expert(weight, raw_type, e, dequant = False):
    # ── 加载并合并所有 expert 的 w1（gate）和 w3（up）──────────────────────────  
    w1_list, w3_list = [], []  
    w1_scale_list, w3_scale_list = [], []  
    w2_list = []  
    w2_scale_list = []  
    for i in range(e):  
        # weight: [N, K//2] I8（nibble packing）  
        w1_list.append(weight[f"layers.3.ffn.experts.{i}.w1.weight"])  
        w3_list.append(weight[f"layers.3.ffn.experts.{i}.w3.weight"])  
        # scale: [N, K//32] uint8（E8M0，即 float8_e8m0fnu 的 raw bytes）  
        w1_scale_list.append(weight[f"layers.3.ffn.experts.{i}.w1.scale"])  
        w3_scale_list.append(weight[f"layers.3.ffn.experts.{i}.w3.scale"])  

        # weight: [K, N//2] I8（nibble packing，K=hidden_size, N=intermediate_size）  
        w2_list.append(weight[f"layers.3.ffn.experts.{i}.w2.weight"])  
        # scale: [K, N//32] uint8（E8M0）  
        w2_scale_list.append(weight[f"layers.3.ffn.experts.{i}.w2.scale"])  

    
    if raw_type=="mxfp4":
        # stack 成 [E, N, K//2]  
        w1_i8 = torch.stack(w1_list, dim=0).view(torch.uint8)   # [E, N, K//2]  
        w3_i8 = torch.stack(w3_list, dim=0).view(torch.uint8)  
        
        w1_scale_e8m0 = torch.stack(w1_scale_list, dim=0).view(torch.uint8)  # [E, N, K//32]  
        w3_scale_e8m0 = torch.stack(w3_scale_list, dim=0).view(torch.uint8)  
        
        # ── 合并 gate+up → w13（vllm 内部格式）────────────────────────────────────  
        w13_i8 = torch.cat([w1_i8, w3_i8], dim=1)              # [E, 2*N, K//2]  
        w13_scale_e8m0 = torch.cat([w1_scale_e8m0, w3_scale_e8m0], dim=1)

        w2_i8 = torch.stack(w2_list, dim=0).view(torch.uint8)          # [E, K, N//2]  
        w2_scale_e8m0 = torch.stack(w2_scale_list, dim=0).view(torch.uint8)
        if dequant:
            E, N2, K_half = w13_i8.shape  
            K = K_half * 2  
            E, K, N_half = w2_i8.shape
            N = N_half * 2  

            w1_fp16 = torch.empty(E, N2, K, dtype=torch.bfloat16, device=w13_i8.device)  
            w2_fp16 = torch.empty(E, K, N, dtype=torch.bfloat16, device=w2_i8.device)
            for i in range(E):  
                w1_fp16[i] = dequantize_mxfp4(w13_i8[i], w13_scale_e8m0[i])
                w2_fp16[i] = dequantize_mxfp4(w2_i8[i], w2_scale_e8m0[i])
        else:
            w1_fp16 = None
            w2_fp16 = None
        return w13_i8, w13_scale_e8m0, w2_i8, w2_scale_e8m0, w1_fp16, w2_fp16
    else:
        # stack 成 [E, N, K//2]  
        w1_fp8 = torch.stack(w1_list, dim=0)   # [E, N, K//2]  
        w3_fp8 = torch.stack(w3_list, dim=0)  
        
        w1_scale_e8m0 = torch.stack(w1_scale_list, dim=0)  # [E, N, K//32]  
        w3_scale_e8m0 = torch.stack(w3_scale_list, dim=0)  
        
        # ── 合并 gate+up → w13（vllm 内部格式）────────────────────────────────────  
        w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1)              # [E, 2*N, K//2]  
        w13_scale_e8m0 = torch.cat([w1_scale_e8m0, w3_scale_e8m0], dim=1).float()

        w2_fp8 = torch.stack(w2_list, dim=0)         # [E, K, N//2]  
        w2_scale_e8m0 = torch.stack(w2_scale_list, dim=0).float()
        if dequant:
            E, N2, K = w13_fp8.shape  
            E, K, N = w2_fp8.shape

            w1_fp16 = torch.empty(E, N2, K, dtype=torch.bfloat16, device=w13_fp8.device)  
            w2_fp16 = torch.empty(E, K, N, dtype=torch.bfloat16, device=w2_fp8.device)
            for i in range(E):  
                w1_fp16[i] = per_block_cast_from_fp8(w13_fp8[i], w13_scale_e8m0[i])
                w2_fp16[i] = per_block_cast_from_fp8(w2_fp8[i], w2_scale_e8m0[i])
        else:
            w1_fp16 = None
            w2_fp16 = None

        return w13_fp8, w13_scale_e8m0, w2_fp8, w2_scale_e8m0, w1_fp16, w2_fp16
    
def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, block_shape):
    """Fused moe with block-wise quantization using native torch."""
    B, D = a.shape
    topk = topk_ids.size(1)
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
            out[mask] = native_w8a8_block_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

E = 256   # num_experts  
M = 6     # token number
N = 2048  # intermediate_size（单个 w1 的 N）  
K = 4096  # hidden_size（真实 K，nibble packing 后 K//2=2048）  
topk = 6  # topk experts per token
block_size = [128, 128]  

weights_mxfp4 = load_file("/root/Data/tmp/raw/model-00005-of-00048.safetensors")  
# generate bfp16 weight from fp8
w13_i8, w13_scale_e8m0, w2_i8, w2_scale_e8m0, w1_fp16, w2_fp16 = load_all_expert(weight=weights_mxfp4, e =E, raw_type="mxfp4", dequant=True)
if True:# load from fp8 file Flash-Base
    # using convert.py
    weights_fp8 = load_file("/root/Data/tmp/save/model0-mp1.safetensors")  
    if True: # generate bfp16 weight from fp8
        w1_fp8, w1_scale, w2_fp8, w2_scale, _, _ = load_all_expert(weight=weights_fp8, e=E, raw_type="fp8", dequant=False)
    else:
        w1_fp8, w1_scale, w2_fp8, w2_scale, w1_fp16, w2_fp16 = load_all_expert(weight=weights_fp8, e=E, raw_type="fp8", dequant=True)
else:
    # 转换权重（w1 是 gate+up 合并，shape [E, 2*N, K]）  
    w1_fp8, w1_scale, w1_fp16 = convert_moe_mxfp4_to_fp8_block(w13_i8, w13_scale_e8m0, block_size)  
    w2_fp8, w2_scale, w2_fp16 = convert_moe_mxfp4_to_fp8_block(w2_i8, w2_scale_e8m0, block_size)  



  
init_workspace_manager(torch.device("cuda:0"))
# 构建 quant_config  
quant_config = fp8_w8a8_moe_quant_config(  
    w1_scale=w1_scale.cuda(),   # [E, 2N//128, K//128] float32  
    w2_scale=w2_scale.cuda(),   # [E, K//128, N//128] float32  
    block_shape=block_size,  
)  
m_fused_moe = modular_triton_fused_moe(make_dummy_moe_config(), quant_config)
  
hidden_states = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) 
score = torch.randn((M, E), device="cuda", dtype=torch.bfloat16)

topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk, False)


# 调用 fused_experts  
out = fused_experts(  
    hidden_states=hidden_states,  # [M, K] bfloat16  
    w1=w1_fp8.cuda(),                    # [E, 2N, K] float8_e4m3fn  
    w2=w2_fp8.cuda(),                    # [E, K, N] float8_e4m3fn  
    topk_weights=topk_weights,    # [M, topk]  
    topk_ids=topk_ids,            # [M, topk]  
    quant_config=quant_config,  
)

try:
    from .utils import opcheck, stack_and_dev, torch_experts, torch_moe
except ImportError:
    from utils import opcheck, stack_and_dev, torch_experts, torch_moe


e_map = None
vllm_config = VllmConfig()
with set_current_vllm_config(vllm_config):
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weights, topk_ids = torch.topk(score, topk)
    torch_output = torch_experts(
        hidden_states,
        w1_fp16.cuda(),
        w2_fp16.cuda(),
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=E,
        expert_map=e_map,
        quant_dtype=torch.bfloat16,
        per_act_token_quant=True,
    )
    ref_out = torch_w8a8_block_fp8_moe(
        hidden_states,
        w1_fp8.cuda(),
        w2_fp8.cuda(),
        quant_config.w1_scale,
        quant_config.w2_scale,
        topk_weights,
        topk_ids,
        block_size,
    )
    m_out = m_fused_moe.apply(
        hidden_states,
        w1_fp8.cuda(),
        w2_fp8.cuda(),
        topk_weights,
        topk_ids,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
        global_num_experts=w1_fp8.shape[0],
    )
# fused_experts_cpu
torch.testing.assert_close(out, torch_output, atol=4e-2, rtol=0)

torch.testing.assert_close(out, ref_out, atol=4e-2, rtol=0)

torch.testing.assert_close(out, m_out, atol=4e-2, rtol=0)




@dataclass
class custom_moe(torch.nn.Module):
    params_dtype: torch.dtype = torch.bfloat16,

layer = custom_moe(params_dtype=torch.bfloat16)
w13_mxfp4_i32, w2_mxfp4_i32, w13_scale_mxfp4_fp8, w2_scale_mxfp4_fp8, w13_bias, w2_bias = prepare_moe_mxfp4_layer_for_marlin(
            layer,
            w13_i8.cuda(),
            w2_i8.cuda(),
            w13_scale_e8m0.cuda(),
            w2_scale_e8m0.cuda(),
            None,
            None,
        )


# mxfp4
marlin_output = fused_marlin_moe(
    hidden_states,
    w13_mxfp4_i32,
    w2_mxfp4_i32,
    None,
    None,
    w13_scale_mxfp4_fp8,
    w2_scale_mxfp4_fp8,
    topk_weights,
    topk_ids,
    global_num_experts=E,
    expert_map=e_map,
    global_scale1=None,
    global_scale2=None,
    g_idx1=None,
    g_idx2=None,
    input_global_scale1=None,
    input_global_scale2=None,
    sort_indices1=None,
    sort_indices2=None,
    w1_zeros=None,
    w2_zeros=None,
    input_dtype=torch.bfloat16,
    quant_type_id=scalar_types.float4_e2m1f.id,
    is_k_full=True,
)
torch.testing.assert_close(marlin_output, torch_output, atol=4e-2, rtol=0)

pass