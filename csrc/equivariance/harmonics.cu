// Real spherical harmonics as a monomial table. One thread owns one vector.
// A term (C, i, j, k, o) contributes
//
//   Y[o] += C * v[i]? * v[j]? * v[k]?
//
// A missing slot is -1. Degree 0 is the constant (every slot -1). Repeated
// component indices are repeated slots, so v[i]^2 contributes 2 v[i] dv[i].
// There is no reduction across vectors and no atomics.
//
// First adjoint (cotangent g of Y):
//   deg 0: nothing
//   deg 1: gv[i] += C*g
//   deg 2: gv[i] += C*v[j]*g     gv[j] += C*v[i]*g
//   deg 3: the product rule over the three slots
//
// Second adjoint (cotangent dgv of gv). dY does not depend on a further
// cotangent of g in this kernel; dg is produced from dgv.
//   deg 1: dg[o] += C*dgv[i]          dv = 0
//   deg 2/3: the product rule. See the kernel.
//
// Accumulation is in the compute dtype. Third derivatives are refused in Python.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace molnex::equivariance {
namespace {

constexpr int kMaxComponents = 16;
constexpr int kBlock = 128;

struct Term {
  int i, j, k, o;
};

__device__ Term load_term(const int16_t* terms, int t) {
  const int16_t* p = terms + static_cast<long long>(t) * 4;
  return {p[0], p[1], p[2], p[3]};
}

template <typename T>
__global__ void forward_kernel(const T* __restrict__ v, const int16_t* __restrict__ terms,
                               const T* __restrict__ coeff, T* __restrict__ y, int n, int components,
                               int nterms) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const T* vr = v + static_cast<long long>(row) * 3;
  T acc[kMaxComponents];
#pragma unroll
  for (int local = 0; local < kMaxComponents; ++local) acc[local] = T(0);
  for (int t = 0; t < nterms; ++t) {
    const Term term = load_term(terms, t);
    T value = coeff[t];
    if (term.i >= 0) value *= vr[term.i];
    if (term.j >= 0) value *= vr[term.j];
    if (term.k >= 0) value *= vr[term.k];
    acc[term.o] += value;
  }
  T* yr = y + static_cast<long long>(row) * components;
  for (int local = 0; local < components; ++local) yr[local] = acc[local];
}

template <typename T>
__global__ void feature_adjoint_kernel(const T* __restrict__ v, const int16_t* __restrict__ terms,
                                        const T* __restrict__ coeff, const T* __restrict__ g,
                                        T* __restrict__ gv, int n, int components, int nterms) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const T* vr = v + static_cast<long long>(row) * 3;
  const T* gr = g + static_cast<long long>(row) * components;
  T acc[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) acc[axis] = T(0);
  for (int t = 0; t < nterms; ++t) {
    const Term term = load_term(terms, t);
    if (term.i < 0) continue;
    const T scale = coeff[t] * gr[term.o];
    if (term.j < 0) {
      acc[term.i] += scale;
    } else if (term.k < 0) {
      acc[term.i] += scale * vr[term.j];
      acc[term.j] += scale * vr[term.i];
    } else {
      acc[term.i] += scale * vr[term.j] * vr[term.k];
      acc[term.j] += scale * vr[term.i] * vr[term.k];
      acc[term.k] += scale * vr[term.i] * vr[term.j];
    }
  }
  T* out = gv + static_cast<long long>(row) * 3;
  out[0] = acc[0];
  out[1] = acc[1];
  out[2] = acc[2];
}

template <typename T>
__global__ void second_kernel(const T* __restrict__ v, const int16_t* __restrict__ terms,
                              const T* __restrict__ coeff, const T* __restrict__ g,
                              const T* __restrict__ dgv, T* __restrict__ dv, T* __restrict__ dg, int n,
                              int components, int nterms) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const T* vr = v + static_cast<long long>(row) * 3;
  const T* gr = g + static_cast<long long>(row) * components;
  const T* dr = dgv + static_cast<long long>(row) * 3;
  T acc[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) acc[axis] = T(0);
  T* dgr = dg + static_cast<long long>(row) * components;
  for (int local = 0; local < components; ++local) dgr[local] = T(0);
  for (int t = 0; t < nterms; ++t) {
    const Term term = load_term(terms, t);
    if (term.i < 0) continue;
    if (term.j < 0) {
      dgr[term.o] += coeff[t] * dr[term.i];
    } else if (term.k < 0) {
      const T base = coeff[t] * gr[term.o];
      acc[term.i] += dr[term.j] * base;
      acc[term.j] += dr[term.i] * base;
      dgr[term.o] += coeff[t] * (dr[term.i] * vr[term.j] + dr[term.j] * vr[term.i]);
    } else {
      const T base = coeff[t] * gr[term.o];
      acc[term.i] += base * (dr[term.j] * vr[term.k] + dr[term.k] * vr[term.j]);
      acc[term.j] += base * (dr[term.i] * vr[term.k] + dr[term.k] * vr[term.i]);
      acc[term.k] += base * (dr[term.i] * vr[term.j] + dr[term.j] * vr[term.i]);
      dgr[term.o] += coeff[t] * (dr[term.i] * vr[term.j] * vr[term.k] +
                                 dr[term.j] * vr[term.i] * vr[term.k] +
                                 dr[term.k] * vr[term.i] * vr[term.j]);
    }
  }
  T* out = dv + static_cast<long long>(row) * 3;
  out[0] = acc[0];
  out[1] = acc[1];
  out[2] = acc[2];
}

void check_vector(const at::Tensor& v) {
  TORCH_CHECK(v.is_cuda(), "spherical harmonics requires a CUDA vector");
  TORCH_CHECK(v.scalar_type() == at::kFloat || v.scalar_type() == at::kDouble,
              "spherical harmonics supports float32 and float64");
  TORCH_CHECK(v.dim() == 2 && v.size(1) == 3 && v.is_contiguous(),
              "spherical harmonics expects a contiguous (N, 3) matrix");
  TORCH_CHECK(v.size(0) <= std::numeric_limits<int>::max(), "too many vectors");
}

void check_terms(const at::Tensor& terms, const at::Tensor& coeff, const at::Tensor& like,
                 int* nterms) {
  TORCH_CHECK(terms.is_cuda() && coeff.is_cuda() && terms.device() == like.device() &&
                  coeff.device() == like.device(),
              "spherical-harmonics term tables must be on the vector device");
  TORCH_CHECK(terms.scalar_type() == at::kShort && terms.is_contiguous() && terms.dim() == 2 &&
                  terms.size(1) == 4,
              "spherical-harmonics terms must be a contiguous int16 (T, 4) table");
  TORCH_CHECK(coeff.scalar_type() == like.scalar_type() && coeff.is_contiguous() && coeff.dim() == 1 &&
                  coeff.numel() == terms.size(0),
              "spherical-harmonics coefficients must match the term count and the vector dtype");
  TORCH_CHECK(terms.size(0) <= std::numeric_limits<int>::max(), "too many terms");
  *nterms = static_cast<int>(terms.size(0));
}

}  // namespace

at::Tensor harmonics_cuda(const at::Tensor& v, const at::Tensor& terms, const at::Tensor& coefficients,
                          int64_t components) {
  const c10::cuda::CUDAGuard guard(v.device());
  check_vector(v);
  TORCH_CHECK(components >= 0 && components <= kMaxComponents, "too many spherical-harmonics components");
  int nterms = 0;
  check_terms(terms, coefficients, v, &nterms);
  const int n = static_cast<int>(v.size(0));
  auto y = at::zeros({v.size(0), components}, v.options());
  if (n == 0 || components == 0 || nterms == 0) return y;
  const int blocks = (n + kBlock - 1) / kBlock;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "spherical_harmonics_forward", [&] {
    forward_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
        v.const_data_ptr<scalar_t>(), terms.const_data_ptr<int16_t>(),
        coefficients.const_data_ptr<scalar_t>(), y.mutable_data_ptr<scalar_t>(), n,
        static_cast<int>(components), nterms);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::tuple<at::Tensor, at::Tensor> harmonics_adjoints_cuda(const at::Tensor& v, const at::Tensor& g,
                                                           const at::Tensor& dgx, const at::Tensor& dgy,
                                                           const at::Tensor& terms,
                                                           const at::Tensor& coefficients, bool second,
                                                           int64_t components) {
  const c10::cuda::CUDAGuard guard(v.device());
  check_vector(v);
  TORCH_CHECK(components >= 0 && components <= kMaxComponents, "too many spherical-harmonics components");
  int nterms = 0;
  check_terms(terms, coefficients, v, &nterms);
  const int n = static_cast<int>(v.size(0));
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == v.scalar_type() && g.is_contiguous() && g.dim() == 2 &&
                  g.size(0) == v.size(0) && g.size(1) == components,
              "cotangent g must be a contiguous (N, components) matrix");
  TORCH_CHECK(dgy.is_cuda() && dgy.scalar_type() == v.scalar_type(),
              "dgy must be a CUDA tensor of the compute dtype");
  auto gv = at::zeros({v.size(0), int64_t{3}}, v.options());
  auto grad_g = at::zeros({second ? v.size(0) : int64_t{0}, components}, v.options());
  if (n == 0 || nterms == 0 || components == 0) return {gv, grad_g};
  if (second) {
    TORCH_CHECK(dgx.sizes() == v.sizes() && dgx.is_contiguous() && dgx.scalar_type() == v.scalar_type() &&
                    dgx.device() == v.device(),
                "the second-derivative cotangent must match the vectors");
  }
  const int blocks = (n + kBlock - 1) / kBlock;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "spherical_harmonics_adjoints", [&] {
    if (!second) {
      feature_adjoint_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
          v.const_data_ptr<scalar_t>(), terms.const_data_ptr<int16_t>(),
          coefficients.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(),
          gv.mutable_data_ptr<scalar_t>(), n, static_cast<int>(components), nterms);
    } else {
      second_kernel<scalar_t><<<blocks, kBlock, 0, stream>>>(
          v.const_data_ptr<scalar_t>(), terms.const_data_ptr<int16_t>(),
          coefficients.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(),
          dgx.const_data_ptr<scalar_t>(), gv.mutable_data_ptr<scalar_t>(),
          grad_g.mutable_data_ptr<scalar_t>(), n, static_cast<int>(components), nterms);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gv, grad_g};
}

}  // namespace molnex::equivariance
