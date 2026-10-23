// Host entry points: validate, pick the (L1, L2) instantiation, launch.
#include <ATen/ATen.h>

#include <algorithm>
#include <limits>

#include <c10/cuda/CUDAGuard.h>

#include "dispatch.cuh"

namespace molnex::equivariance {
namespace {

struct Shape {
  int64_t na, nb, np, nk, nt;
};

// The same enumeration as Config<L1, L2>, evaluated at runtime for validation.
Shape shape_of(int64_t l1, int64_t l2) {
#define MOLNEX_EQUIVARIANCE_SHAPE(A, B) \
  if (l1 == A && l2 == B)               \
    return {Config<A, B>::NA, Config<A, B>::NB, Config<A, B>::NP, Config<A, B>::NK, Config<A, B>::NT};
  MOLNEX_EQUIVARIANCE_CONFIGS(MOLNEX_EQUIVARIANCE_SHAPE)
#undef MOLNEX_EQUIVARIANCE_SHAPE
  TORCH_CHECK(false, "unsupported (l1, l2) = (", l1, ", ", l2, "): l1 in 0..3, l2 in 2..3");
}

void check_operand(const at::Tensor& x, const at::Tensor& like, int64_t rows, int64_t width,
                   const char* name) {
  TORCH_CHECK(x.device() == like.device() && x.scalar_type() == like.scalar_type(),
              name, " must share the device and dtype of the first operand");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(1) == width && (rows < 0 || x.size(0) == rows),
              name, " must be a contiguous (", rows < 0 ? -1 : rows, ", ", width, ") matrix, got ",
              x.sizes());
}

void check_index(const at::Tensor& x, const at::Tensor& like, int64_t numel, at::ScalarType type,
                 const char* name) {
  TORCH_CHECK(x.device() == like.device() && x.scalar_type() == type && x.dim() == 1 &&
                  x.is_contiguous() && x.numel() == numel,
              name, " must be a contiguous ", type, " vector of length ", numel, ", got ",
              x.sizes(), " ", x.scalar_type());
}

void check_common(const at::Tensor& a, const Shape& s, int64_t channels, const at::Tensor& b,
                  const at::Tensor& w, const at::Tensor& source, const at::Tensor& target,
                  at::ArrayRef<double> coefficients) {
  TORCH_CHECK(a.is_cuda(), "operands must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kDouble,
              "channelwise supports float32/float64 only");
  TORCH_CHECK(channels == 32 || channels == 64 || channels == 128,
              "unsupported channel multiplicity ", channels);
  const int64_t edges = source.numel();
  check_operand(a, a, -1, s.na * channels, "a");
  check_operand(b, a, edges, s.nb, "b");
  check_operand(w, a, edges, s.np * channels, "w");
  check_index(source, a, edges, at::kLong, "source");
  check_index(target, a, edges, at::kLong, "target");
  TORCH_CHECK(coefficients.size() == size_t(s.nt), "coefficient table must have ", s.nt, " entries");
}

// Sum the ``splits`` partial results of a row-split kernel; a view when there is one.
at::Tensor reduce_partials(const at::Tensor& partial) {
  const int64_t splits = partial.size(0), numel = partial.numel() / std::max<int64_t>(splits, 1);
  if (splits == 1) return partial[0];
  auto out = at::empty({partial.size(1), partial.size(2)}, partial.options());
  if (numel == 0) return out;
  AT_DISPATCH_FLOATING_TYPES(partial.scalar_type(), "channelwise_reduce_partials", [&] {
    reduce_partials_kernel<scalar_t><<<unsigned((numel + 255) / 256), 256, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
        partial.const_data_ptr<scalar_t>(), out.mutable_data_ptr<scalar_t>(), splits, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

at::Tensor segment_sum(const at::Tensor& values, const at::Tensor& order, const at::Tensor& rowptr,
                       int64_t rows) {
  auto out = at::empty({rows, values.size(1)}, values.options());
  if (rows == 0) return out;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many rows for one grid");
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "channelwise_segment_sum", [&] {
    segment_sum_kernel<scalar_t><<<unsigned(rows), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        values.const_data_ptr<scalar_t>(), order.const_data_ptr<int32_t>(),
        rowptr.const_data_ptr<int32_t>(), out.mutable_data_ptr<scalar_t>(), values.size(1),
        values.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

}  // namespace

at::Tensor product_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& w,
                        const at::Tensor& source, const at::Tensor& target,
                        const at::Tensor& order_t, const at::Tensor& rowptr_t,
                        const at::Tensor&, const at::Tensor&, at::ArrayRef<double> coefficients,
                        int64_t rows, int64_t l1, int64_t l2, int64_t channels) {
  const auto s = shape_of(l1, l2);
  const c10::cuda::CUDAGuard guard(a.device());
  check_common(a, s, channels, b, w, source, target, coefficients);
  TORCH_CHECK(rows >= 0, "rows must be non-negative");
  check_index(order_t, a, source.numel(), at::kInt, "order_t");
  check_index(rowptr_t, a, rows + 1, at::kInt, "rowptr_t");
  const int64_t splits = row_splits(rows);
  auto out = at::empty({splits, rows, s.nk * channels}, a.options());
  ProductArgs x{a, b, w, source, order_t, rowptr_t, out, coefficients, channels, splits};
#define MOLNEX_EQUIVARIANCE_RUN(A, B) \
  if (l1 == A && l2 == B) run_product<A, B>(x);
  MOLNEX_EQUIVARIANCE_CONFIGS(MOLNEX_EQUIVARIANCE_RUN)
#undef MOLNEX_EQUIVARIANCE_RUN
  return reduce_partials(out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> adjoints_cuda(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& w, const at::Tensor& g,
    const at::Tensor& da, const at::Tensor& db, const at::Tensor& dw, const at::Tensor& source,
    const at::Tensor& target, const at::Tensor& order_t, const at::Tensor& rowptr_t,
    const at::Tensor& order_s, const at::Tensor& rowptr_s, at::ArrayRef<double> coefficients,
    bool second, int64_t l1, int64_t l2, int64_t channels) {
  const auto s = shape_of(l1, l2);
  const c10::cuda::CUDAGuard guard(a.device());
  check_common(a, s, channels, b, w, source, target, coefficients);
  const int64_t edges = source.numel(), rows_a = a.size(0);
  check_operand(g, a, -1, s.nk * channels, "g");
  const int64_t rows_d = g.size(0);
  if (second) {
    check_operand(da, a, rows_a, s.na * channels, "da");
    check_operand(db, a, edges, s.nb, "db");
    check_operand(dw, a, edges, s.np * channels, "dw");
  }
  check_index(order_t, a, edges, at::kInt, "order_t");
  check_index(rowptr_t, a, rows_d + 1, at::kInt, "rowptr_t");
  check_index(order_s, a, edges, at::kInt, "order_s");
  check_index(rowptr_s, a, rows_a + 1, at::kInt, "rowptr_s");
  auto ga_edge = at::empty({edges, s.na * channels}, a.options());
  auto gb = at::empty({edges, s.nb}, a.options());
  auto gw = at::empty({edges, s.np * channels}, a.options());
  const int64_t splits = row_splits(rows_d);
  auto grad_g = at::empty({splits, second ? rows_d : 0, s.nk * channels}, a.options());
  AdjointArgs x{a,  b,  w,  g,  da, db, dw, source, order_t, rowptr_t, ga_edge, gb, gw, grad_g,
                coefficients, channels, second, splits};
#define MOLNEX_EQUIVARIANCE_RUN(A, B) \
  if (l1 == A && l2 == B) run_adjoint<A, B>(x);
  MOLNEX_EQUIVARIANCE_CONFIGS(MOLNEX_EQUIVARIANCE_RUN)
#undef MOLNEX_EQUIVARIANCE_RUN
  return {segment_sum(ga_edge, order_s, rowptr_s, rows_a), gb, gw, reduce_partials(grad_g)};
}

}  // namespace molnex::equivariance
