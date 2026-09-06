// CPU implementation of `molix::get_neighbor_pairs`.
//
// O(N^2) pair enumeration. Fast path: all validation is the caller's
// responsibility (done in the Python wrapper). The only filtering done
// here is algorithmic:
//   * distance > cutoff  → drop (outside interaction range)
//   * distance == 0      → drop (self-interaction; caller's data error,
//                                 but filtering avoids NaN in backward)

#include <torch/extension.h>
#include <tuple>

using std::tuple;
using torch::arange;
using torch::div;
using torch::full;
using torch::hstack;
using torch::index_select;
using torch::indexing::Slice;
using torch::kDouble;
using torch::kInt32;
using torch::kInt64;
using torch::linalg_vector_norm;
using torch::outer;
using torch::round;
using torch::Scalar;
using torch::Tensor;
using torch::vstack;

static tuple<Tensor, Tensor, Tensor, Tensor> forward(
    const Tensor& positions,
    const Scalar& cutoff,
    const Scalar& max_num_pairs,
    const Tensor& box_vectors
) {
    const int max_num_pairs_ = max_num_pairs.to<int>();
    const int num_atoms = positions.size(0);
    const int64_t num_all_pairs = static_cast<int64_t>(num_atoms) * (num_atoms - 1) / 2;

    // 64-bit for the same reason as the CUDA kernel: `8 * index` leaves int32
    // from ~23 200 atoms on, and the float sqrt cannot resolve it either.
    const Tensor indices = arange(0, num_all_pairs, positions.options().dtype(kInt64));
    Tensor rows64 = (((8 * indices + 1).to(kDouble).sqrt() + 1) / 2).floor().to(kInt64);
    rows64 -= (rows64 * (rows64 - 1) > 2 * indices).to(kInt64);
    const Tensor columns64 = indices - div(rows64 * (rows64 - 1), 2, "floor");
    const Tensor rows = rows64.to(kInt32);
    const Tensor columns = columns64.to(kInt32);

    Tensor neighbors = vstack({rows, columns});
    Tensor deltas = index_select(positions, 0, rows64) - index_select(positions, 0, columns64);
    if (box_vectors.size(0) != 0) {
        deltas -= outer(round(deltas.index({Slice(), 2}) / box_vectors.index({2, 2})), box_vectors.index({2}));
        deltas -= outer(round(deltas.index({Slice(), 1}) / box_vectors.index({1, 1})), box_vectors.index({1}));
        deltas -= outer(round(deltas.index({Slice(), 0}) / box_vectors.index({0, 0})), box_vectors.index({0}));
    }
    Tensor distances = linalg_vector_norm(deltas, 2, 1);

    const Tensor valid = (distances <= cutoff) & (distances > 0);
    const int num_found = valid.sum().item<int>();

    if (max_num_pairs_ == -1) {
        const Tensor mask = ~valid;
        neighbors.index_put_({Slice(), mask}, -1);
        deltas = deltas.clone(); // Break an autograd loop
        distances = distances.clone();
        deltas.index_put_({mask, Slice()}, NAN);
        distances.index_put_({mask}, NAN);
    } else {
        neighbors = neighbors.index({Slice(), valid});
        deltas = deltas.index({valid, Slice()});
        distances = distances.index({valid});

        const int num_pad = max_num_pairs_ - num_found;
        if (num_pad > 0) {
            neighbors = hstack({neighbors, full({2, num_pad}, -1, neighbors.options())});
            deltas = vstack({deltas, full({num_pad, 3}, NAN, deltas.options())});
            distances = hstack({distances, full({num_pad}, NAN, distances.options())});
        }
    }
    Tensor num_pairs_found = torch::empty(1, indices.options().dtype(kInt32));
    num_pairs_found[0] = num_found;
    return {neighbors, deltas, distances, num_pairs_found};
}

TORCH_LIBRARY_IMPL(molix, AutogradCPU, m) {
    m.impl("get_neighbor_pairs", &forward);
}
