#ifndef MOLIX_ACCESSOR_H
#define MOLIX_ACCESSOR_H

#include <torch/extension.h>

// Packed 32-bit accessor with `__restrict__` pointer traits for use in
// kernels. 32-bit indices are a deliberate trade-off: sufficient for every
// per-tensor byte offset we ever hit (< 2^31 elements) and materially
// faster than 64-bit on device.
template <typename scalar_t, int num_dims>
using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dims, torch::RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const torch::Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
}

#endif  // MOLIX_ACCESSOR_H
