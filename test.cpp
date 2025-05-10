#include <hip/hip_runtime.h>
__device__ uint8_t  test_fp4(float a, float b, float scale) {
            const int clamp = 0;
            const int round = 0;
            // This should compile into v_cvt_scalef32_pk_fp4_f32
            uint8_t r = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(a, b, scale, clamp ,round);
            //asm volatile("" :: "r"(r)); // prevent optimization
           return r;
}

__global__ void dummy_kernel(float* out) {
            out[0] = test_fp4(1.0f, 2.0f, 1.0f);
}
