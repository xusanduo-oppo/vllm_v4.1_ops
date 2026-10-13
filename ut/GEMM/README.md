TODO
DeepGemmFp8BlockScaledMMKernel/DeepGemmMxfp8BmmLinearKernel, fp8_gemm_nt
MarlinMxfp8LinearKernel/MarlinFP8ScaledMMLinearKernel

layers.0.attn.wq_b(2) 		
                                                                       repeat             
layers.0.attn.wq_b.scale	[1 024, 40]	           TP4        [256, 40] =====> [8 192, 40]       prepare_mxfp8_layer_for_marlin             [40, 8 192]
F8_E8M0                                           =====>                                         ==============================>             

layers.0.attn.wq_b.weight	[32 768, 1 280]	                  [8 192, 1 280]                                                                [80, 32 768]
F8_E4M3