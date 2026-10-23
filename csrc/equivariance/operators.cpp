// Schemas and Meta kernels of the independent channelwise library.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>

namespace molnex::equivariance {
at::Tensor product_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        at::ArrayRef<double>, int64_t, int64_t, int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> adjoints_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::ArrayRef<double>, bool, int64_t, int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule_cuda(const at::Tensor&,
                                                                          const at::Tensor&,
                                                                          int64_t, int64_t);

at::Tensor product_meta(const at::Tensor& a, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        at::ArrayRef<double>, int64_t rows, int64_t l1, int64_t l2,
                        int64_t channels) {
  // output rows per channel = Σ_paths (2·l3+1); paths·channels = w.size(1)
  int64_t nk = 0;
  for (int64_t x1 = 0; x1 <= l1; ++x1)
    for (int64_t x2 = 0; x2 <= l2; ++x2)
      for (int64_t x3 = std::abs(x1 - x2); x3 <= std::min(x1 + x2, l2); ++x3)
        if ((x1 + x2 + x3) % 2 == 0) nk += 2 * x3 + 1;
  return at::empty({rows, nk * channels}, a.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> adjoints_meta(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& w, const at::Tensor& g,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor& source,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::ArrayRef<double>, bool second, int64_t, int64_t, int64_t) {
  const int64_t edges = source.numel();
  return {at::empty({a.size(0), a.size(1)}, a.options()), at::empty({edges, b.size(1)}, a.options()),
          at::empty({edges, w.size(1)}, a.options()),
          at::empty({second ? g.size(0) : 0, g.size(1)}, a.options())};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule_meta(
    const at::Tensor& source, const at::Tensor&, int64_t rows_a, int64_t rows_d) {
  const auto options = source.options().dtype(at::kInt);
  return {at::empty({source.numel()}, options), at::empty({rows_d + 1}, options),
          at::empty({source.numel()}, options), at::empty({rows_a + 1}, options)};
}
}  // namespace molnex::equivariance

TORCH_LIBRARY(molnex_equivariance, m) {
  // The product at the target rows over the edge schedule by target.
  // The by-source schedule is carried, unused, so the adjoints have it.
  m.def("product(Tensor a, Tensor b, Tensor w, Tensor source, Tensor target, "
        "Tensor order_t, Tensor rowptr_t, Tensor order_s, Tensor rowptr_s, "
        "float[] coefficients, int rows, int l1, int l2, int channels) -> Tensor");
  // The adjoints (ga, gb, gw) for the cotangent g — and, with second=True and
  // the cotangents (da, db, dw) of those, the adjoints of the adjoints
  // (grad_a, grad_b, grad_w, grad_g). With second=False, da/db/dw are ignored
  // and grad_g is empty.
  m.def("adjoints(Tensor a, Tensor b, Tensor w, Tensor g, Tensor da, Tensor db, Tensor dw, "
        "Tensor source, Tensor target, Tensor order_t, Tensor rowptr_t, Tensor order_s, "
        "Tensor rowptr_s, float[] coefficients, bool second, int l1, int l2, int channels) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
  // CSR of the edge list by target and by source: (order_t, rowptr_t, order_s, rowptr_s).
  m.def("schedule(Tensor source, Tensor target, int rows_a, int rows_d) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(molnex_equivariance, CUDA, m) {
  m.impl("product", molnex::equivariance::product_cuda);
  m.impl("adjoints", molnex::equivariance::adjoints_cuda);
  m.impl("schedule", molnex::equivariance::schedule_cuda);
}
TORCH_LIBRARY_IMPL(molnex_equivariance, Meta, m) {
  m.impl("product", molnex::equivariance::product_meta);
  m.impl("adjoints", molnex::equivariance::adjoints_meta);
  m.impl("schedule", molnex::equivariance::schedule_meta);
}
