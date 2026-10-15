// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// firefly Skillet (NVFP4 SM75 decode): NVFP4 权重 → m16n8k8 tensor-core decode GEMM。
//
// 背景: NVFP4 的 scale 是两级——每 16 个 k 一个 fp8_e4m3 group scale + 一个
// fp32 global scale。上游 Marlin 每个 decode step 都要在寄存器里做 E4M3 位解码
// 并把两级 scale 交织进 mma 累加。本 kernel 把 E4M3×global 在**加载期 prepack**
// 时预解成 fp16 per-group scale, decode 时权重 E2M1 寄存器现解、乘预解好的
// fp16 scale 直接进 mma, 消除每 step 的 E4M3 解码。
//
// 架构: SM75 (Turing, T10) 没有 k16 fp16 mma, 只有 m16n8k8。做 m16n8k16 等效
// 须发**两条 m16n8k8** (与上游 Marlin sm75 分支一致)。tile 是 16 行 N × 8 列 k
// (每指令), 不用 m8n8k4 (那是更老架构的 tile, sm75 浪费 tensor core)。
//
// 张量:
//   weight:      uint8    [N, K/2]    行主序, 每 byte = 2 E2M1 (低 4bit=k 偶, 高 4bit=k 奇)
//   group_scale: fp8_e4m3 [N, K/16]   每 16 个 k 一个 scale
//   global_scale: fp32 标量
//
// 反量化: w_deq[n,k] = E2M1(nibble[n,k]) * E4M3(group_scale[n,k//16]) * global_scale
//
// 三个算子 (独立 torch extension, namespace firefly_skillet, 不污染 vllm._C):
//   prepare(weight, scale, global) -> [codes, scales]
//       codes  uint8 [N, K/2]  E2M1 nibble 重排成 m16n8k8 B 操作数 fragment 顺序
//       scales fp16  [N, K/16] E4M3×global 预解成 fp16 (RNE round), 核心优化
//   unpack(codes) -> weight_packed uint8[N, K/2]
//       prepare 里 codes 重排的精确逆 (prefill 解回行主序喂 nvfp4_to_int8_cached)
//   gemm(out, x, codes, scales)
//       decode GEMM: out[m,n] = sum_k x[m,k]*w_deq[n,k], m16n8k8 mma, M∈[1,16], split-K
//
// m16n8k8 片段布局 (mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32, 单指令 k=8):
//   group g = lane>>2 (0..7), tid = lane&3 (0..3)
//   A (16x8, 2 u32):  a0 → row=g   的 k 列 {2*tid, 2*tid+1}
//                     a1 → row=g+8 的 k 列 {2*tid, 2*tid+1}
//   B (8x8, 1 u32):   b0 → col=g   的 k 行 {2*tid, 2*tid+1}  (B 是 k 维连续, n 是 col)
//   C (16x8, 4 f32):  c0 → (row=g,   col=2*tid)
//                     c1 → (row=g,   col=2*tid+1)
//                     c2 → (row=g+8, col=2*tid)
//                     c3 → (row=g+8, col=2*tid+1)
//   m16n8k16 = 两条 k8: 第 1 条 A={a0,a1} (k[0,8)) B={b 前半},
//                       第 2 条 A={a0',a1'} (k[8,16)) B={b 后半}, 累加同一 C。
//
// E2M1→fp16 (纯位操作, 免 LUT): E2M1 nibble 的 3 位幅值直接放进 fp16 的
// 指数+尾数低位, 得到值 = E2M1/16384 (fp16 精确); 寄存器里 ×16384 还原成
// E2M1 (power of 2, 无舍入), 再乘预解 fp16 scale (单次 RNE round) 即权重。
// 8 个 E2M1 幅值 {0,0.5,1,1.5,2,3,4,6} 都落在 fp16 normal 区, 无 subnormal/溢出。

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

// E2M1 一个 nibble (4bit) → fp16 位模式, 值 = E2M1(nib)/16384 (精确, 无舍入)。
// 纯位操作: sign(bit3)→fp16 sign(bit15), 幅值 3 位(bit2..0)→fp16 指数低 2 位+尾数高位。
__device__ __forceinline__ unsigned e2m1_fp16_bits(unsigned nib) {
  nib &= 0xFu;
  return ((nib & 0x8u) << 12) | ((nib & 0x7u) << 9);
}

// 单个 E2M1 nibble → half (权重值 = E2M1 × 预解 fp16 scale, 单次 RNE round)。
// 纯位操作免 LUT: nibble 位模式 = E2M1/16384 (fp16 精确), ×16384 还原再乘 scale。
__device__ __forceinline__ half e2m1_nibble_to_half(uint8_t nib, half scale) {
  const half code = __ushort_as_half(static_cast<unsigned short>(e2m1_fp16_bits(nib)));
  const half k16 = __float2half_rn(16384.0f);  // power of 2, 精确还原 E2M1
  return __hmul(__hmul(code, k16), scale);
}

// 一个 byte (2 个 E2M1 nibble) → half2, 各 nibble 乘预解 fp16 scale (scale 两 half 相同)。
// lo = 低 nibble, hi = 高 nibble。
__device__ __forceinline__ half2 dequant_byte_to_half2(uint8_t byte, half2 scale) {
  const half s = __low2half(scale);
  return __halves2half2(e2m1_nibble_to_half(byte & 0xFu, s),
                        e2m1_nibble_to_half((byte >> 4) & 0xFu, s));
}

// m16n8k8 f16 → f32 mma, 单指令 k=8。A 操作数 = 2 个 b32 寄存器 (4 half):
//   第 1 个 b32 = a0 (m 行 0..7 的 k8), 第 2 个 b32 = a1 (m 行 8..15 的 k8)。
// B = 1 个 b32 (2 half, n 列, k8)。C = 4 float (m16n8)。
// 注: a0/a1 各是一个 half2 (2 half), 整个 A = {a0, a1} 共 4 half, 不能只喂 a0
// (只喂 a0 时 m 行 8..15 缺失, M>8 会 NaN/错值)。
__device__ __forceinline__ void mma_m16n8k8_f16(float* c, const half2 a0,
                                                const half2 a1, const half2 b0) {
  const unsigned a0r = *reinterpret_cast<const unsigned*>(&a0);  // m 行 0..7
  const unsigned a1r = *reinterpret_cast<const unsigned*>(&a1);  // m 行 8..15
  const unsigned b0r = *reinterpret_cast<const unsigned*>(&b0);
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0r), "r"(a1r), "r"(b0r), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]));
}

// ---------------------------------------------------------------------------
// prepare-codes (宽 load 布局): 把 weight 行主序 byte 重排成 m16n8k8 B 操作数
// fragment 顺序, 且让**同一 lane 的 4 个连续 group 相邻 (16B)**, 供 decode kernel
// 每 lane 一次 uint4=16B 读 4 组 (对标 Marlin cp.async4 的 16B/lane, LDG 指令数
// ÷4, DRAM bank 并行更好 → gu 207→259GB/s)。
// 布局 codes[tile][gc][lane] = uint4 (16B), gc = group/4 (K%64==0 → groups%4==0
// 布局永远整), uint4[gin] = 组 g=gc*4+gin 的 4 byte {n0·k8lo, n0·k8hi, n1·k8lo,
// n1·k8hi}。固定 (tile, gc) 时 32 lane 读 [lane*16, lane*16+16) → 512B=4
// cache-line 无过取 (每 16B=1 sector)。与 unpack_kernel 互为逆 (byte 逐位一致)。
// 地址公式 (forward, tile=blockIdx.x, gc=blockIdx.y, lane=threadIdx.x):
//   uint4 pv[gin] (g=gc*4+gin) 的 4 byte:
//     b[0] = weight[(tile*16 + quad)*half_k + g*8 + tid]
//     b[1] = weight[(tile*16 + quad)*half_k + g*8 + 4 + tid]
//     b[2] = weight[(tile*16 + 8+quad)*half_k + g*8 + tid]
//     b[3] = weight[(tile*16 + 8+quad)*half_k + g*8 + 4 + tid]
//   写入 codes[(tile*(K/64) + gc)*32 + lane] 的 16B。quad=lane>>2, tid=lane&3。
// ---------------------------------------------------------------------------
__global__ void prepare_codes_kernel(
    const uint8_t* __restrict__ weight,  // [N, K/2] 行主序
    uint8_t* __restrict__ codes,         // [N, K/2] (宽布局, 同字节数重排)
    int N, int K) {
  const int tile = blockIdx.x;          // N/16 tile
  const int gc = blockIdx.y;            // K/64 chunk (4 组)
  const int lane = threadIdx.x;         // 0..31
  const int quad = lane >> 2;
  const int tid = lane & 3;
  const int half_k = K >> 1;
  const int n0 = tile * 16 + quad;
  const int n1 = tile * 16 + 8 + quad;
  uint4 out;
  uint32_t* pv = reinterpret_cast<uint32_t*>(&out);
  #pragma unroll
  for (int gin = 0; gin < 4; ++gin) {
    const int g = gc * 4 + gin;
    uint32_t v;
    uint8_t* b = reinterpret_cast<uint8_t*>(&v);
    b[0] = weight[static_cast<size_t>(n0) * half_k + g * 8 + tid];
    b[1] = weight[static_cast<size_t>(n0) * half_k + g * 8 + 4 + tid];
    b[2] = weight[static_cast<size_t>(n1) * half_k + g * 8 + tid];
    b[3] = weight[static_cast<size_t>(n1) * half_k + g * 8 + 4 + tid];
    pv[gin] = v;
  }
  *reinterpret_cast<uint4*>(
      codes + (static_cast<size_t>(tile * (K / 64) + gc) * 32 + lane) * 16) = out;
}

// ---------------------------------------------------------------------------
// prepare-scales: E4M3 × global → fp16 (RNE round), 存 [N, K/16] 行主序。
// 核心优化: decode 时不再解 E4M3, 直接用这个预解 fp16 scale。
// ---------------------------------------------------------------------------
__global__ void prepare_scales_kernel(
    const uint8_t* __restrict__ group_scale,  // fp8_e4m3 bytes [N, K/16]
    float global_scale,                       // host 标量
    half* __restrict__ scales,                // fp16 [N, K/16]
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int byte = group_scale[idx];
  const int sign = (byte >> 7) & 1;
  const int exp = (byte >> 3) & 0xF;
  const int man = byte & 0x7;
  float val;
  if (exp == 0) {
    val = static_cast<float>(man) * 0.125f * 0.015625f;  // subnormal: man/8 * 2^-6
  } else if (exp == 15 && man == 7) {
    val = 0.0f;  // 0x7F = e4m3fn NaN, 正常 group scale 不出现, 按 0 处理
  } else {
    val = ldexpf(static_cast<float>(man) * 0.125f + 1.0f, exp - 7);
  }
  scales[idx] = __float2half_rn(sign ? -val * global_scale : val * global_scale);
}

// ---------------------------------------------------------------------------
// unpack: prepare-codes 宽布局的精确逆 → 还原行主序 weight [N, K/2]。
// 每 (tile, gc, lane) 读 16B (4 组), 拆回行主序 4 byte/组, 与 prepare forward
// 逐 byte 对应 (prefill 用它把 codes 反解回行主序喂 nvfp4_to_int8_cached)。
// ---------------------------------------------------------------------------
__global__ void unpack_kernel(
    const uint8_t* __restrict__ codes,  // [N, K/2] 宽布局
    uint8_t* __restrict__ weight,       // [N, K/2] 行主序
    int N, int K) {
  const int tile = blockIdx.x;
  const int gc = blockIdx.y;  // K/64 chunk
  const int lane = threadIdx.x;
  const int quad = lane >> 2;
  const int tid = lane & 3;
  const int half_k = K >> 1;
  const int n0 = tile * 16 + quad;
  const int n1 = tile * 16 + 8 + quad;
  const uint4 w = *reinterpret_cast<const uint4*>(
      codes + (static_cast<size_t>(tile * (K / 64) + gc) * 32 + lane) * 16);
  const uint32_t* pv = reinterpret_cast<const uint32_t*>(&w);
  #pragma unroll
  for (int gin = 0; gin < 4; ++gin) {
    const int g = gc * 4 + gin;
    const uint8_t* bb = reinterpret_cast<const uint8_t*>(&pv[gin]);
    weight[static_cast<size_t>(n0) * half_k + g * 8 + tid] = bb[0];
    weight[static_cast<size_t>(n0) * half_k + g * 8 + 4 + tid] = bb[1];
    weight[static_cast<size_t>(n1) * half_k + g * 8 + tid] = bb[2];
    weight[static_cast<size_t>(n1) * half_k + g * 8 + 4 + tid] = bb[3];
  }
}

// ---------------------------------------------------------------------------
// gemm: decode GEMM, m16n8k8 mma, M∈[1,16], split-K。
// 每 block = 1 个 16 行 N tile (blockIdx.x), SPK warp 把 K/16 group 切分 (split-K),
// 各 warp 累加 C 片段后写 shared partial[SPK][16][16] 归约, 再 round fp16 写 out。
// A 片段: 行 = quad & quad+8 (即 m 行), k8-half0 → a0/a1, k8-half1 → a0'/a1'。
// B 片段: 每 (weight row, k8-half) 解 1 byte 成 half2, 乘预解 fp16 scale。
// C 片段: c0..c3 → (row quad/quad+8, col 2*tid/2*tid+1); n0 走 0..7, n1 走 8..15。
//
// SPK 是 split-K 因子 = block 内 warp 数 (= block 线程数 / 32)。SM75 该 kernel 是
// **延迟受限** (mma 占 ~3%, 带宽占 ~7%), 非带宽 bound: 增大 SPK → 每 block 独立
// mma 链更多, 延迟隐藏更好, m=1 提速 (T10 实测 gu 4352x5120 spk8→spk16 快 30%)。
// 归约只由前 256 线程做 (输出固定 16x16=256 元素), SPK*32>256 时其余线程空转,
// 避免 partial[SPK][16][16] 越界 (mrow 只 0..15 合法)。
// ---------------------------------------------------------------------------
template <int SPK>
__global__ __launch_bounds__(SPK * 32) void gemm_kernel(
    const half* __restrict__ x,       // [M, K] fp16 行主序
    const uint8_t* __restrict__ codes, // [N, K/2]
    const half* __restrict__ scales,   // fp16 [N, K/16]
    half* __restrict__ out,           // [M, N] fp16
    int M, int N, int K) {
  const int tile = blockIdx.x;
  const int groups = K >> 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = lane >> 2;
  const int tid = lane & 3;
  const int k_local0 = tid * 2;  // k8-half0 内 2 列偏移

  // split-K: SPK warp 切分 group [0, groups)
  const int g0 = warp * groups / SPK;
  const int g1 = (warp + 1) * groups / SPK;

  // C 片段: n0 (weight 行 0..7) 与 n1 (weight 行 8..15), 各 4 float
  float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int n0 = tile * 16 + quad;
  const int n1 = tile * 16 + 8 + quad;

  // 宽 load (同 gemm_m1_kernel): codes[tile][gc][lane] = uint4 (16B), 每 lane 一次
  // 读 4 组 (LDG ÷4)。warp 段 [g0,g1) → 全局 chunk [g0>>2, (g1+3)>>2), 边界 chunk
  // 只处理段内组 (mask)。
  const int gchunks = groups >> 2;  // K%64==0 → groups%4==0
  const uint4* cbase = reinterpret_cast<const uint4*>(
      codes + static_cast<size_t>(tile) * gchunks * 32 * 16);
  const int gc0 = g0 >> 2;
  const int gc1 = (g1 + 3) >> 2;

  for (int gc = gc0; gc < gc1; ++gc) {
    const uint4 w = cbase[static_cast<size_t>(gc) * 32 + lane];  // 4 组, 16B/lane
    const uint32_t cb[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int gin = 0; gin < 4; ++gin) {
      const int g = gc * 4 + gin;
      if (g < g0 || g >= g1) continue;  // 边界 chunk 非本 warp 的组跳过
      const int kabs = g * 16 + k_local0;  // 全局 k (k8-half0 起点)
      // A 片段 (row = quad / quad+8, k8-half0 → a0/a1, k8-half1 → a2/a3)
      half ax_q0, ax_q1, ax_q8_0, ax_q8_1, ax_q0b, ax_q1b, ax_q8_0b, ax_q8_1b;
      if (quad < M) {
        ax_q0 = x[static_cast<size_t>(quad) * K + kabs];
        ax_q1 = x[static_cast<size_t>(quad) * K + kabs + 1];
        ax_q0b = x[static_cast<size_t>(quad) * K + kabs + 8];
        ax_q1b = x[static_cast<size_t>(quad) * K + kabs + 9];
      } else {
        ax_q0 = ax_q1 = ax_q0b = ax_q1b = __float2half(0.0f);
      }
      if (quad + 8 < M) {
        ax_q8_0 = x[static_cast<size_t>(quad + 8) * K + kabs];
        ax_q8_1 = x[static_cast<size_t>(quad + 8) * K + kabs + 1];
        ax_q8_0b = x[static_cast<size_t>(quad + 8) * K + kabs + 8];
        ax_q8_1b = x[static_cast<size_t>(quad + 8) * K + kabs + 9];
      } else {
        ax_q8_0 = ax_q8_1 = ax_q8_0b = ax_q8_1b = __float2half(0.0f);
      }
      const half2 a0 = __halves2half2(ax_q0, ax_q1);
      const half2 a1 = __halves2half2(ax_q8_0, ax_q8_1);
      const half2 a2 = __halves2half2(ax_q0b, ax_q1b);
      const half2 a3 = __halves2half2(ax_q8_0b, ax_q8_1b);

      // B: 该 lane 的这 1 组 4 byte → 解 E2M1 × 预解 fp16 scale。
      // scale 转置布局 [K/16, N]: 固定 (tile,g) 时 32 lane 读 n 连续 → 1 cache-line。
      const uint32_t cb4 = cb[gin];
      const half s0 = scales[static_cast<size_t>(g) * N + n0];
      const half s1 = scales[static_cast<size_t>(g) * N + n1];
      const half2 scale0 = __halves2half2(s0, s0);
      const half2 scale1 = __halves2half2(s1, s1);
      const half2 b0 = dequant_byte_to_half2(cb4 & 0xFFu, scale0);         // n0, k8-half0
      const half2 b1 = dequant_byte_to_half2((cb4 >> 8) & 0xFFu, scale0);  // n0, k8-half1
      const half2 b2 = dequant_byte_to_half2((cb4 >> 16) & 0xFFu, scale1); // n1, k8-half0
      const half2 b3 = dequant_byte_to_half2((cb4 >> 24) & 0xFFu, scale1); // n1, k8-half1

      // n0 mma (2 条 k8): 累加 c1; n1 mma (2 条 k8): 累加 c2
      mma_m16n8k8_f16(c1, a0, a1, b0);
      mma_m16n8k8_f16(c1, a2, a3, b1);
      mma_m16n8k8_f16(c2, a0, a1, b2);
      mma_m16n8k8_f16(c2, a2, a3, b3);
    }
  }

  // C 片段写 shared [SPK warp][16 row][16 col], 归约 SPK warp
  __shared__ float partial[SPK][16][16];
  // c1 (n0, col 0..7): c0→(quad, 2*tid), c1→(quad, 2*tid+1), c2→(quad+8, 2*tid), c3→(quad+8, 2*tid+1)
  partial[warp][quad][tid * 2 + 0] = c1[0];
  partial[warp][quad][tid * 2 + 1] = c1[1];
  partial[warp][quad + 8][tid * 2 + 0] = c1[2];
  partial[warp][quad + 8][tid * 2 + 1] = c1[3];
  // c2 (n1, col 8..15)
  partial[warp][quad][8 + tid * 2 + 0] = c2[0];
  partial[warp][quad][8 + tid * 2 + 1] = c2[1];
  partial[warp][quad + 8][8 + tid * 2 + 0] = c2[2];
  partial[warp][quad + 8][8 + tid * 2 + 1] = c2[3];
  __syncthreads();

  // 输出固定 16 行 m × 16 列 n = 256 元素, 只由前 256 线程归约。SPK*32>256
  // (如 SPK=16 → 512 线程) 时其余线程空转; 必须 guard, 否则 mrow=e>>4 越界 partial。
  if (threadIdx.x < 256) {
    const int mrow = threadIdx.x >> 4;   // 激活行 m, 0..15
    const int col = threadIdx.x & 15;    // 权重列 (tile 内), 0..15
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < SPK; ++w) acc += partial[w][mrow][col];
    if (mrow < M) {
      out[static_cast<size_t>(mrow) * N + tile * 16 + col] = __float2half_rn(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// gemm_m1: m=1 decode 专化 kernel。m=1 时 m16 tile 的 row 1..15 是零填充垃圾,
// 永不被下游读; 但通用 gemm_kernel 仍写/归约全部 16 行 (每 block 16x16 shared,
// 16KB 写 + 16KB 读 + syncthreads)。gu(N=8704) 有 544 block → ~17MB shared 流量,
// 是 Marlin (内部 N-split, 56 block) 的 36µs 差之主要来源 (in-context 实测
// Marlin gu 79us vs Skillet 115us, 差额在归约而非 DRAM 加载)。
//
// 本 kernel: 只 quad==0 (row 0) 的 lane 写 partial[warp][col], 归约只 16 线程,
// shared 从 [SPK][16][16] 砍到 [SPK][16] (流量 ÷16)。mma 仍跑 16 行 (硬件约束),
// 但死行 A=0 不读 x (省 15 次 x 全局读)。bit-exact (bench_m1.py 验证 True)。
// gu 170→122us (1.39x), dn 60→58us (1.03x) standalone; in-context 待 e2e 验。
// ---------------------------------------------------------------------------
template <int SPK>
__global__ __launch_bounds__(SPK * 32) void gemm_m1_kernel(
    const half* __restrict__ x,        // [1, K] fp16
    const uint8_t* __restrict__ codes,  // [N, K/2]
    const half* __restrict__ scales,    // fp16 [N, K/16]
    half* __restrict__ out,            // [1, N] fp16
    int N, int K) {
  const int tile = blockIdx.x;
  const int groups = K >> 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = lane >> 2;
  const int tid = lane & 3;
  const int k_local0 = tid * 2;

  const int g0 = warp * groups / SPK;
  const int g1 = (warp + 1) * groups / SPK;

  float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int n0 = tile * 16 + quad;
  const int n1 = tile * 16 + 8 + quad;

  const half zero = __float2half(0.0f);
  const half2 z2 = __halves2half2(zero, zero);

  // 宽 load (对标 Marlin cp.async4 的 16B/lane): codes 宽布局 codes[tile][gc][lane]
  // = uint4 (16B), 每 lane 一次读 4 组 (LDG 指令数 ÷4, DRAM bank 并行更好)。gu
  // 207→259GB/s, gu 107.5→85.9µs (1.25x), dn 57.9→55.5µs (1.04x), bit-exact。
  // warp 段 [g0,g1) 对应全局 chunk [g0>>2, (g1+3)>>2); 边界 chunk 只处理落在段内
  // 的组 (mask)。gu (g0/g1 是 20 的倍数) 对齐零浪费; dn (17 组/warp) 边界浪费但正确。
  const int gchunks = groups >> 2;  // K%64==0 → groups%4==0
  const uint4* cbase = reinterpret_cast<const uint4*>(
      codes + static_cast<size_t>(tile) * gchunks * 32 * 16);
  const int gc0 = g0 >> 2;
  const int gc1 = (g1 + 3) >> 2;

  for (int gc = gc0; gc < gc1; ++gc) {
    const uint4 w = cbase[static_cast<size_t>(gc) * 32 + lane];  // 4 组, 16B/lane
    const uint32_t cb[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int gin = 0; gin < 4; ++gin) {
      const int gg = gc * 4 + gin;
      if (gg < g0 || gg >= g1) continue;  // 边界 chunk 非本 warp 的组跳过
      const int kabs = gg * 16 + k_local0;
      half2 a0, a1, a2, a3;
      if (quad == 0) {
        a0 = __halves2half2(x[kabs], x[kabs + 1]);       // row0 k8-half0
        a2 = __halves2half2(x[kabs + 8], x[kabs + 9]);   // row0 k8-half1
        a1 = a3 = z2;                                     // 死行 (m=1 无 row 8..15)
      } else {
        a0 = a1 = a2 = a3 = z2;
      }
      const half s0 = scales[static_cast<size_t>(gg) * N + n0];
      const half s1 = scales[static_cast<size_t>(gg) * N + n1];
      const uint32_t cb4 = cb[gin];
      const half2 scale0 = __halves2half2(s0, s0);
      const half2 scale1 = __halves2half2(s1, s1);
      const half2 b0 = dequant_byte_to_half2(cb4 & 0xFFu, scale0);
      const half2 b1 = dequant_byte_to_half2((cb4 >> 8) & 0xFFu, scale0);
      const half2 b2 = dequant_byte_to_half2((cb4 >> 16) & 0xFFu, scale1);
      const half2 b3 = dequant_byte_to_half2((cb4 >> 24) & 0xFFu, scale1);
      mma_m16n8k8_f16(c1, a0, a1, b0);
      mma_m16n8k8_f16(c1, a2, a3, b1);
      mma_m16n8k8_f16(c2, a0, a1, b2);
      mma_m16n8k8_f16(c2, a2, a3, b3);
    }
  }

  // row-0 only: c1 → row0 col 0..7, c2 → row0 col 8..15, 均在 quad==0。
  __shared__ float partial[SPK][16];
  if (quad == 0) {
    partial[warp][tid * 2 + 0] = c1[0];
    partial[warp][tid * 2 + 1] = c1[1];
    partial[warp][8 + tid * 2 + 0] = c2[0];
    partial[warp][8 + tid * 2 + 1] = c2[1];
  }
  __syncthreads();
  if (threadIdx.x < 16) {
    const int col = threadIdx.x;
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < SPK; ++w) acc += partial[w][col];
    out[tile * 16 + col] = __float2half_rn(acc);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// host 入口
// ---------------------------------------------------------------------------

// prepare: codes uint8[N,K/2], scales fp16[N,K/16]
std::vector<torch::Tensor> skillet_prepare(torch::Tensor weight_packed,
                                        torch::Tensor weight_scale,
                                        double global_scale) {
  TORCH_CHECK(weight_packed.is_cuda() && weight_scale.is_cuda(),
              "skillet_prepare 需 CUDA 张量");
  TORCH_CHECK(weight_packed.scalar_type() == torch::kUInt8,
              "skillet_prepare: weight 需 uint8");
  TORCH_CHECK(weight_scale.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "skillet_prepare: scale 需 fp8_e4m3fn");
  TORCH_CHECK(weight_packed.dim() == 2 && weight_scale.dim() == 2,
              "skillet_prepare: 需 2D");
  TORCH_CHECK(weight_packed.is_contiguous() && weight_scale.is_contiguous(),
              "skillet_prepare: 需 contiguous");
  TORCH_CHECK(weight_packed.get_device() == weight_scale.get_device(),
              "skillet_prepare: 需同设备");
  const int64_t N = weight_packed.size(0);
  const int64_t K = weight_packed.size(1) * 2;
  TORCH_CHECK(N % 16 == 0, "skillet_prepare: N%16==0 (m16 tile), got N=", N);
  TORCH_CHECK(K % 64 == 0,
              "skillet_prepare: K%64==0 (宽 load 布局 4 组/chunk), got K=", K);
  TORCH_CHECK(weight_scale.size(0) == N && weight_scale.size(1) == K / 16,
              "skillet_prepare: scale shape 应为 [N, K/16]");

  auto codes = torch::empty_like(weight_packed);
  auto scales =
      torch::empty({N, K / 16}, weight_packed.options().dtype(torch::kFloat16));

  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  // 宽 load 布局: 每 block 处理 1 tile × 1 chunk (4 组), grid.y = K/64。
  const dim3 grid(static_cast<unsigned>(N / 16), static_cast<unsigned>(K / 64));
  prepare_codes_kernel<<<grid, 32, 0, stream>>>(
      weight_packed.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(),
      static_cast<int>(N), static_cast<int>(K));
  const int total = static_cast<int>(N * (K / 16));
  const int blocks = (total + 255) / 256;
  prepare_scales_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<float>(global_scale),
      reinterpret_cast<half*>(scales.data_ptr()), total);
  // 转置成 [K/16, N]: decode kernel 固定 (tile,g) 时按 n 连续读 scale (1 cache-line),
  // 行主序 [N, K/16] 里 n 隔 groups 行 → ~16 cache-line/组, gu 下 scale 取流量大。
  // 一次性 (加载期) 转置, 加载成本可忽略。返回后 kernel 全部按 [K/16, N] 读。
  auto scales_t = scales.t().contiguous();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales_t};
}

torch::Tensor skillet_unpack(torch::Tensor codes) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kUInt8,
              "skillet_unpack: 需 CUDA uint8");
  TORCH_CHECK(codes.dim() == 2, "skillet_unpack: 需 2D");
  TORCH_CHECK(codes.is_contiguous(), "skillet_unpack: 需 contiguous");
  const int64_t N = codes.size(0);
  const int64_t K = codes.size(1) * 2;
  TORCH_CHECK(N % 16 == 0 && K % 64 == 0,
              "skillet_unpack: N%16==0, K%64==0 (宽 load 布局需 4 组/chunk)");
  auto weight = torch::empty_like(codes);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned>(N / 16), static_cast<unsigned>(K / 64));
  unpack_kernel<<<grid, 32, 0, stream>>>(codes.data_ptr<uint8_t>(),
                                         weight.data_ptr<uint8_t>(),
                                         static_cast<int>(N), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return weight;
}

void skillet_gemm(torch::Tensor out, torch::Tensor x, torch::Tensor codes,
               torch::Tensor scales, int64_t spk) {
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && codes.is_cuda() && scales.is_cuda(),
              "skillet_gemm: 需全 CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  x.scalar_type() == torch::kFloat16,
              "skillet_gemm: out/x 需 fp16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kFloat16,
              "skillet_gemm: codes 需 uint8, scales 需 fp16");
  TORCH_CHECK(out.dim() == 2 && x.dim() == 2, "skillet_gemm: 需 2D");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  codes.is_contiguous() && scales.is_contiguous(),
              "skillet_gemm: 需 contiguous");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = out.size(1);
  TORCH_CHECK(out.size(0) == M, "skillet_gemm: out 行数需等于 x 行数");
  TORCH_CHECK(M >= 1 && M <= 16, "skillet_gemm: M∈[1,16], got M=", M);
  TORCH_CHECK(N % 16 == 0 && K % 64 == 0,
              "skillet_gemm: N%16==0, K%64==0 (宽 load 布局), got K=", K);
  TORCH_CHECK(codes.size(0) == N && codes.size(1) == K / 2,
              "skillet_gemm: codes 形状应为 [N, K/2]");
  // scales 是转置布局 [K/16, N] (prepare 时 .t().contiguous())。
  TORCH_CHECK(scales.size(0) == K / 16 && scales.size(1) == N,
              "skillet_gemm: scales 形状应为 [K/16, N] (转置)");

  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned>(N / 16));
  const half* xx = reinterpret_cast<const half*>(x.data_ptr());
  const uint8_t* cc = codes.data_ptr<uint8_t>();
  const half* ss = reinterpret_cast<const half*>(scales.data_ptr());
  half* oo = reinterpret_cast<half*>(out.data_ptr());
  // split-K 因子 (block 内 warp 数)。支持 2/4/8/16, 其余回退 8。
  // 各模板实例独立编译; 只实例化用到的 (spk 运行期选定, 编译期全编)。
  // m=1 decode 走 row-0-only 专化 kernel (归约流量 ÷16), M>1 走通用 kernel。
  if (M == 1) {
    switch (spk) {
      case 2:
        gemm_m1_kernel<2><<<grid, 64, 0, stream>>>(xx, cc, ss, oo, (int)N, (int)K);
        break;
      case 4:
        gemm_m1_kernel<4><<<grid, 128, 0, stream>>>(xx, cc, ss, oo, (int)N, (int)K);
        break;
      case 16:
        gemm_m1_kernel<16><<<grid, 512, 0, stream>>>(xx, cc, ss, oo, (int)N, (int)K);
        break;
      case 8:
      default:
        gemm_m1_kernel<8><<<grid, 256, 0, stream>>>(xx, cc, ss, oo, (int)N, (int)K);
        break;
    }
  } else {
    switch (spk) {
      case 2:
        gemm_kernel<2><<<grid, 64, 0, stream>>>(xx, cc, ss, oo, (int)M, (int)N, (int)K);
        break;
      case 4:
        gemm_kernel<4><<<grid, 128, 0, stream>>>(xx, cc, ss, oo, (int)M, (int)N, (int)K);
        break;
      case 16:
        gemm_kernel<16><<<grid, 512, 0, stream>>>(xx, cc, ss, oo, (int)M, (int)N, (int)K);
        break;
      case 8:
      default:
        gemm_kernel<8><<<grid, 256, 0, stream>>>(xx, cc, ss, oo, (int)M, (int)N, (int)K);
        break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("prepare", &skillet_prepare, "Skillet prepack (codes, scales)");
  m.def("unpack", &skillet_unpack, "Skillet codes -> 行主序 weight (prepare 的逆)");
  m.def("gemm", &skillet_gemm, "NVFP4 Skillet decode GEMM (out 就地写)");
}
