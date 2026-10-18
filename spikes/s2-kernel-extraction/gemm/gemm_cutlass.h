#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int roost_s2_gemm_cutlass_128x128x32(const void* a, const void* w, void* c, int m, int n, int k,
                                     void* ws, void* stream);
int roost_s2_gemm_cutlass_32x64x64(const void* a, const void* w, void* c, int m, int n, int k,
                                   void* ws, void* stream);
#ifdef __cplusplus
}
#endif
