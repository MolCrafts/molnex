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
at::Tensor contraction_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            at::ArrayRef<double>, int64_t, int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor, at::Tensor> contraction_adjoints_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::ArrayRef<double>, int64_t, bool, int64_t, int64_t);
at::Tensor contraction_table(int64_t, int64_t, int64_t);
at::Tensor harmonics_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t);
std::tuple<at::Tensor, at::Tensor> harmonics_adjoints_cuda(const at::Tensor&, const at::Tensor&,
                                                           const at::Tensor&, const at::Tensor&,
                                                           const at::Tensor&, const at::Tensor&, bool,
                                                           int64_t);
at::Tensor contracter_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, at::ArrayRef<double>, at::ArrayRef<double>, int64_t,
                           int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> contracter_adjoints_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::ArrayRef<double>, at::ArrayRef<double>, int64_t, bool, int64_t, int64_t);
at::Tensor contracter_table(int64_t);
at::Tensor weighted_channels_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor, at::Tensor> weighted_channels_adjoints_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, int64_t, int64_t,
    int64_t);

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

at::Tensor contraction_meta(const at::Tensor& x, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            at::ArrayRef<double>, int64_t, int64_t channels, int64_t segments_out) {
  return at::empty({x.size(0), segments_out * channels}, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> contraction_adjoints_meta(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, at::ArrayRef<double>, int64_t, bool second,
    int64_t channels, int64_t segments_out) {
  const int64_t width = segments_out * channels;
  return {at::empty({x.size(0), x.size(1)}, x.options()), at::empty_like(w),
          at::empty({second ? x.size(0) : int64_t{0}, width}, x.options())};
}

at::Tensor harmonics_meta(const at::Tensor& v, const at::Tensor&, const at::Tensor&,
                          int64_t components) {
  return at::empty({v.size(0), components}, v.options());
}

std::tuple<at::Tensor, at::Tensor> harmonics_adjoints_meta(const at::Tensor& v, const at::Tensor&,
                                                           const at::Tensor&, const at::Tensor&,
                                                           const at::Tensor&, const at::Tensor&,
                                                           bool second, int64_t components) {
  return {at::empty({v.size(0), int64_t{3}}, v.options()),
          at::empty({second ? v.size(0) : int64_t{0}, components}, v.options())};
}

at::Tensor contracter_meta(const at::Tensor& x1, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&, at::ArrayRef<double>,
                           at::ArrayRef<double>, int64_t, int64_t channels, int64_t segments_out) {
  return at::empty({x1.size(0), segments_out * channels}, x1.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> contracter_adjoints_meta(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& w, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::ArrayRef<double>, at::ArrayRef<double>, int64_t, bool second,
    int64_t channels, int64_t segments_out) {
  const int64_t width = segments_out * channels;
  return {at::empty_like(x1), at::empty_like(x2), at::empty_like(w),
          at::empty({second ? x1.size(0) : int64_t{0}, width}, x1.options())};
}

at::Tensor weighted_channels_meta(const at::Tensor& basis, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, const at::Tensor&, int64_t channels, int64_t,
                                  int64_t num_rows) {
  const int64_t rows = num_rows < 0 ? basis.size(0) : num_rows;
  return at::empty({rows, basis.size(1) * channels}, basis.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> weighted_channels_adjoints_meta(
    const at::Tensor& basis, const at::Tensor& weight, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    bool second, int64_t channels, int64_t, int64_t num_rows) {
  const int64_t rows = num_rows < 0 ? basis.size(0) : num_rows;
  const int64_t width = basis.size(1) * channels;
  return {at::empty({basis.size(0), basis.size(1)}, basis.options()), at::empty_like(weight),
          at::empty({second ? rows : int64_t{0}, width}, basis.options())};
}
}  // namespace molnex::equivariance

// The requirement string molnex's deploy metadata records and the C++ runtime
// (interface/runtime OpLibraries) checks against before loading a package.
extern "C" const char* molnex_op_library_requirement() { return "molnex_equivariance==1"; }

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
  // Symmetric contraction. Grouped term tables are inputs so autograd can save
  // them; the forward kernel reads the path-order tables and ignores the groups.
  // ``contraction_adjoints`` with second=False returns (gx, gw, empty). With
  // second=True the same outputs are (dx, dw, grad_g).
  // Symmetric contraction: ``packed`` / ``coefficients`` sorted by output
  // segment (``out_offsets``) for the generic kernel; ``values`` the same
  // coefficients as a host list and ``config`` the compiled table
  // (l_in·100 + l_out·10 + degree, or -1) for the compiled kernels.
  m.def("contraction(Tensor x, Tensor w, Tensor indices, Tensor packed, Tensor coefficients, "
        "Tensor out_offsets, float[] values, int config, int channels, int segments_out) -> Tensor");
  m.def("contraction_adjoints(Tensor x, Tensor w, Tensor indices, Tensor g, Tensor dgx, "
        "Tensor dgw, Tensor dgy, Tensor packed, Tensor coefficients, Tensor out_offsets, "
        "float[] values, int config, bool second, int channels, int segments_out) "
        "-> (Tensor, Tensor, Tensor)");
  // The compiled term table of a configuration, (NT, 5) int16 on the CPU;
  // empty when none is compiled.
  m.def("contraction_table(int l_in, int l_out, int degree) -> Tensor");
  // Spherical harmonics of an (N, 3) matrix. second=False returns (gv, empty).
  // second=True returns (dv, grad_g).
  m.def("harmonics(Tensor v, Tensor terms, Tensor coefficients, int components) -> Tensor");
  m.def("harmonics_adjoints(Tensor v, Tensor g, Tensor dgx, Tensor dgy, Tensor terms, "
        "Tensor coefficients, bool second, int components) -> (Tensor, Tensor)");
  // Allegro contraction. x2 is gathered by index. The weight is one shared row.
  // second=False returns (gx1, gx2, gw, empty). second=True returns
  // (dx1, dx2, dw, grad_g).
  // Allegro contracter over a path table (P, 4): (class id, off1, off2, off3)
  // and the flat Clebsch-Gordan coefficient table of cg.hpp (652 values).
  // ``values`` are the flattened term coefficients and ``config`` the compiled
  // table of contracter_tables.hpp (an id, or -1 for the class-blocked kernel).
  m.def("contracter(Tensor x1, Tensor x2, Tensor w, Tensor index, Tensor paths, float[] cg, "
        "float[] values, int config, int channels, int segments_out) -> Tensor");
  m.def("contracter_adjoints(Tensor x1, Tensor x2, Tensor w, Tensor index, Tensor g, "
        "Tensor dgx1, Tensor dgx2, Tensor dgw, Tensor dgy, Tensor paths, float[] cg, "
        "float[] values, int config, bool second, int channels, int segments_out) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
  // The compiled term table ``id``, (NT, 4) int16 on the CPU; empty beyond the last.
  m.def("contracter_table(int id) -> Tensor");
  m.def("weighted_channels(Tensor basis, Tensor weight, Tensor index, Tensor terms, "
        "Tensor coefficients, int channels, int num_irreps, int num_rows) -> Tensor");
  m.def("weighted_channels_adjoints(Tensor basis, Tensor weight, Tensor index, Tensor g, "
        "Tensor dga, Tensor dgw, Tensor dgy, Tensor terms, Tensor coefficients, bool second, "
        "int channels, int num_irreps, int num_rows) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(molnex_equivariance, CUDA, m) {
  m.impl("product", molnex::equivariance::product_cuda);
  m.impl("adjoints", molnex::equivariance::adjoints_cuda);
  m.impl("schedule", molnex::equivariance::schedule_cuda);
  m.impl("contraction", molnex::equivariance::contraction_cuda);
  m.impl("contraction_adjoints", molnex::equivariance::contraction_adjoints_cuda);
  m.impl("harmonics", molnex::equivariance::harmonics_cuda);
  m.impl("harmonics_adjoints", molnex::equivariance::harmonics_adjoints_cuda);
  m.impl("contracter", molnex::equivariance::contracter_cuda);
  m.impl("contracter_adjoints", molnex::equivariance::contracter_adjoints_cuda);
  m.impl("weighted_channels", molnex::equivariance::weighted_channels_cuda);
  m.impl("weighted_channels_adjoints", molnex::equivariance::weighted_channels_adjoints_cuda);
}
TORCH_LIBRARY_IMPL(molnex_equivariance, CompositeExplicitAutograd, m) {
  m.impl("contraction_table", molnex::equivariance::contraction_table);
  m.impl("contracter_table", molnex::equivariance::contracter_table);
}

TORCH_LIBRARY_IMPL(molnex_equivariance, Meta, m) {
  m.impl("product", molnex::equivariance::product_meta);
  m.impl("adjoints", molnex::equivariance::adjoints_meta);
  m.impl("schedule", molnex::equivariance::schedule_meta);
  m.impl("contraction", molnex::equivariance::contraction_meta);
  m.impl("contraction_adjoints", molnex::equivariance::contraction_adjoints_meta);
  m.impl("harmonics", molnex::equivariance::harmonics_meta);
  m.impl("harmonics_adjoints", molnex::equivariance::harmonics_adjoints_meta);
  m.impl("contracter", molnex::equivariance::contracter_meta);
  m.impl("contracter_adjoints", molnex::equivariance::contracter_adjoints_meta);
  m.impl("weighted_channels", molnex::equivariance::weighted_channels_meta);
  m.impl("weighted_channels_adjoints", molnex::equivariance::weighted_channels_adjoints_meta);
}
