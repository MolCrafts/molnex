// CUDA implementation of PME direct/reciprocal ops.
//
// Dtype-generic via AT_DISPATCH_V2 over {float, double}. Kernels are
// doubly templated (scalar_t, PME_ORDER) so the PME-order specialization
// remains a compile-time constant even after dtype dispatch.

#include <ATen/Dispatch_v2.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <array>
#include <cmath>

#include "molix/accessor.cuh"

using namespace torch::autograd;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;

namespace {

// ── device helpers ─────────────────────────────────────────────────────────

template <typename scalar_t>
__device__ inline void invertBoxVectors(
    const Accessor<scalar_t, 2>& box, scalar_t recipBoxVectors[3][3]
) {
    scalar_t determinant = box[0][0] * box[1][1] * box[2][2];
    scalar_t scale = scalar_t(1) / determinant;
    recipBoxVectors[0][0] = box[1][1] * box[2][2] * scale;
    recipBoxVectors[0][1] = 0;
    recipBoxVectors[0][2] = 0;
    recipBoxVectors[1][0] = -box[1][0] * box[2][2] * scale;
    recipBoxVectors[1][1] = box[0][0] * box[2][2] * scale;
    recipBoxVectors[1][2] = 0;
    recipBoxVectors[2][0] = (box[1][0] * box[2][1] - box[1][1] * box[2][0]) * scale;
    recipBoxVectors[2][1] = -box[0][0] * box[2][1] * scale;
    recipBoxVectors[2][2] = box[0][0] * box[1][1] * scale;
}

template <typename scalar_t>
__device__ inline void computeSpline(
    int atom,
    const Accessor<scalar_t, 2>& pos,
    const Accessor<scalar_t, 2>& box,
    const scalar_t recipBoxVectors[3][3],
    const int gridSize[3],
    int gridIndex[3],
    scalar_t data[][3],
    scalar_t ddata[][3],
    int pmeOrder
) {
    scalar_t posInBox[3] = {pos[atom][0], pos[atom][1], pos[atom][2]};
    for (int i = 2; i >= 0; i--) {
        scalar_t scale = floor(posInBox[i] * recipBoxVectors[i][i]);
        for (int j = 0; j < 3; j++)
            posInBox[j] -= scale * box[i][j];
    }
    scalar_t t[3], dr[3];
    int ti[3];
    for (int i = 0; i < 3; i++) {
        t[i] = posInBox[0] * recipBoxVectors[0][i]
             + posInBox[1] * recipBoxVectors[1][i]
             + posInBox[2] * recipBoxVectors[2][i];
        t[i] = (t[i] - floor(t[i])) * gridSize[i];
        ti[i] = (int)t[i];
        dr[i] = t[i] - ti[i];
        gridIndex[i] = ti[i] % gridSize[i];
    }
    scalar_t scale = scalar_t(1) / (pmeOrder - 1);
    for (int i = 0; i < 3; i++) {
        data[pmeOrder - 1][i] = 0;
        data[1][i] = dr[i];
        data[0][i] = 1 - dr[i];
        for (int j = 3; j < pmeOrder; j++) {
            scalar_t div = scalar_t(1) / (j - 1);
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for (int k = 1; k < j - 1; k++)
                data[j - k - 1][i] = div * ((dr[i] + k) * data[j - k - 2][i]
                                          + (j - k - dr[i]) * data[j - k - 1][i]);
            data[0][i] = div * (1 - dr[i]) * data[0][i];
        }
        if (ddata != nullptr) {
            ddata[0][i] = -data[0][i];
            for (int j = 1; j < pmeOrder; j++)
                ddata[j][i] = data[j - 1][i] - data[j][i];
        }
        data[pmeOrder - 1][i] = scale * dr[i] * data[pmeOrder - 2][i];
        for (int j = 1; j < pmeOrder - 1; j++)
            data[pmeOrder - j - 1][i] = scale * ((dr[i] + j) * data[pmeOrder - j - 2][i]
                                               + (pmeOrder - j - dr[i]) * data[pmeOrder - j - 1][i]);
        data[0][i] = scale * (1 - dr[i]) * data[0][i];
    }
}

static int getMaxBlocks() {
    int device, numMultiprocessors;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device);
    return numMultiprocessors * 4;
}

// ── direct kernel ──────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void pme_direct_cuda_kernel(
    const Accessor<scalar_t, 2> pos,
    const Accessor<scalar_t, 1> charge,
    const Accessor<int, 2> neighbors,
    const Accessor<scalar_t, 2> deltas,
    const Accessor<scalar_t, 1> distances,
    const Accessor<int, 2> exclusions,
    Accessor<scalar_t, 2> posDeriv,
    Accessor<scalar_t, 1> chargeDeriv,
    Accessor<scalar_t, 1> energyBuffer,
    scalar_t alpha, scalar_t coulomb
) {
    int numAtoms = pos.size(0);
    int numNeighbors = neighbors.size(1);
    int maxExclusions = exclusions.size(1);
    scalar_t energy = 0;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numNeighbors;
         index += blockDim.x * gridDim.x) {
        int atom1 = neighbors[0][index];
        int atom2 = neighbors[1][index];
        scalar_t r = distances[index];
        bool include = (atom1 > -1);
        for (int j = 0; include && j < maxExclusions && exclusions[atom1][j] >= atom2; j++)
            if (exclusions[atom1][j] == atom2) include = false;
        if (!include) continue;

        scalar_t invR = 1 / r;
        scalar_t alphaR = alpha * r;
        scalar_t expAlphaRSqr = exp(-alphaR * alphaR);
        scalar_t erfcAlphaR = erfc(alphaR);
        scalar_t prefactor = coulomb * invR;
        scalar_t c1 = charge[atom1];
        scalar_t c2 = charge[atom2];
        energy += prefactor * erfcAlphaR * c1 * c2;
        atomicAdd(&chargeDeriv[atom1], prefactor * erfcAlphaR * c2);
        atomicAdd(&chargeDeriv[atom2], prefactor * erfcAlphaR * c1);
        scalar_t dEdR = prefactor * c1 * c2
                      * (erfcAlphaR + alphaR * expAlphaRSqr * M_2_SQRTPI)
                      * invR * invR;
        for (int j = 0; j < 3; j++) {
            atomicAdd(&posDeriv[atom1][j], -dEdR * deltas[index][j]);
            atomicAdd(&posDeriv[atom2][j], dEdR * deltas[index][j]);
        }
    }

    scalar_t dr[3];
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numAtoms * maxExclusions;
         index += blockDim.x * gridDim.x) {
        int atom1 = index / maxExclusions;
        int atom2 = exclusions[atom1][index - atom1 * maxExclusions];
        if (atom2 > atom1) {
            for (int j = 0; j < 3; j++)
                dr[j] = pos[atom1][j] - pos[atom2][j];
            scalar_t r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            scalar_t invR = rsqrt(r2);
            scalar_t r = invR * r2;
            scalar_t alphaR = alpha * r;
            scalar_t expAlphaRSqr = exp(-alphaR * alphaR);
            scalar_t erfAlphaR = erf(alphaR);
            scalar_t prefactor = coulomb * invR;
            scalar_t c1 = charge[atom1];
            scalar_t c2 = charge[atom2];
            energy -= prefactor * erfAlphaR * c1 * c2;
            atomicAdd(&chargeDeriv[atom1], -prefactor * erfAlphaR * c2);
            atomicAdd(&chargeDeriv[atom2], -prefactor * erfAlphaR * c1);
            scalar_t dEdR = prefactor * c1 * c2
                          * (erfAlphaR - alphaR * expAlphaRSqr * M_2_SQRTPI)
                          * invR * invR;
            for (int j = 0; j < 3; j++) {
                atomicAdd(&posDeriv[atom1][j], dEdR * dr[j]);
                atomicAdd(&posDeriv[atom2][j], -dEdR * dr[j]);
            }
        }
    }
    energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] = energy;
}

// ── reciprocal kernels ─────────────────────────────────────────────────────

template <typename scalar_t, int PME_ORDER>
__global__ void spreadCharge(
    const Accessor<scalar_t, 2> pos,
    const Accessor<scalar_t, 1> charge,
    const Accessor<scalar_t, 2> box,
    Accessor<scalar_t, 3> grid,
    int gridx, int gridy, int gridz, scalar_t sqrtCoulomb
) {
    __shared__ scalar_t recipBoxVectors[3][3];
    if (threadIdx.x == 0) invertBoxVectors<scalar_t>(box, recipBoxVectors);
    __syncthreads();
    scalar_t data[PME_ORDER][3];
    int numAtoms = pos.size(0);
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < numAtoms;
         atom += blockDim.x * gridDim.x) {
        int gridIndex[3];
        int gridSize[3] = {gridx, gridy, gridz};
        computeSpline<scalar_t>(atom, pos, box, recipBoxVectors, gridSize, gridIndex,
                                data, (scalar_t(*)[3])nullptr, PME_ORDER);
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = gridIndex[0] + ix;
            xindex -= (xindex >= gridx ? gridx : 0);
            scalar_t dx = charge[atom] * sqrtCoulomb * data[ix][0];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = gridIndex[1] + iy;
                yindex -= (yindex >= gridy ? gridy : 0);
                scalar_t dxdy = dx * data[iy][1];
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex[2] + iz;
                    zindex -= (zindex >= gridz ? gridz : 0);
                    atomicAdd(&grid[xindex][yindex][zindex], dxdy * data[iz][2]);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void reciprocalConvolution(
    const Accessor<scalar_t, 2> box,
    Accessor<c10::complex<scalar_t>, 3> grid,
    int gridx, int gridy, int gridz,
    const Accessor<scalar_t, 1> xmoduli,
    const Accessor<scalar_t, 1> ymoduli,
    const Accessor<scalar_t, 1> zmoduli,
    scalar_t recipExpFactor,
    Accessor<scalar_t, 1> energyBuffer
) {
    scalar_t recipBoxVectors[3][3];
    invertBoxVectors<scalar_t>(box, recipBoxVectors);
    const unsigned int gridSize = gridx * gridy * (gridz / 2 + 1);
    const scalar_t recipScaleFactor = recipBoxVectors[0][0] * recipBoxVectors[1][1]
                                    * recipBoxVectors[2][2] / scalar_t(M_PI);
    scalar_t energy = 0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < gridSize;
         index += blockDim.x * gridDim.x) {
        int kx = index / (gridy * (gridz / 2 + 1));
        int remainder = index - kx * gridy * (gridz / 2 + 1);
        int ky = remainder / (gridz / 2 + 1);
        int kz = remainder - ky * (gridz / 2 + 1);
        int mx = (kx < (gridx + 1) / 2) ? kx : (kx - gridx);
        int my = (ky < (gridy + 1) / 2) ? ky : (ky - gridy);
        int mz = (kz < (gridz + 1) / 2) ? kz : (kz - gridz);
        scalar_t mhx = mx * recipBoxVectors[0][0];
        scalar_t mhy = mx * recipBoxVectors[1][0] + my * recipBoxVectors[1][1];
        scalar_t mhz = mx * recipBoxVectors[2][0]
                     + my * recipBoxVectors[2][1]
                     + mz * recipBoxVectors[2][2];
        scalar_t bx = xmoduli[kx];
        scalar_t by = ymoduli[ky];
        scalar_t bz = zmoduli[kz];
        c10::complex<scalar_t>& g = grid[kx][ky][kz];
        scalar_t m2 = mhx * mhx + mhy * mhy + mhz * mhz;
        scalar_t denom = m2 * bx * by * bz;
        scalar_t eterm = (index == 0 ? scalar_t(0) : recipScaleFactor * exp(-recipExpFactor * m2) / denom);
        scalar_t scale = (kz > 0 && kz <= (gridz - 1) / 2 ? scalar_t(2) : scalar_t(1));
        energy += scale * eterm * (g.real() * g.real() + g.imag() * g.imag());
        g *= eterm;
    }
    energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] = scalar_t(0.5) * energy;
}

template <typename scalar_t, int PME_ORDER>
__global__ void interpolateForce(
    const Accessor<scalar_t, 2> pos,
    const Accessor<scalar_t, 1> charge,
    const Accessor<scalar_t, 2> box,
    const Accessor<scalar_t, 3> grid,
    int gridx, int gridy, int gridz, scalar_t sqrtCoulomb,
    Accessor<scalar_t, 2> posDeriv,
    Accessor<scalar_t, 1> chargeDeriv
) {
    __shared__ scalar_t recipBoxVectors[3][3];
    if (threadIdx.x == 0) invertBoxVectors<scalar_t>(box, recipBoxVectors);
    __syncthreads();
    scalar_t data[PME_ORDER][3];
    scalar_t ddata[PME_ORDER][3];
    int numAtoms = pos.size(0);
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < numAtoms;
         atom += blockDim.x * gridDim.x) {
        int gridIndex[3];
        int gridSize[3] = {gridx, gridy, gridz};
        computeSpline<scalar_t>(atom, pos, box, recipBoxVectors, gridSize, gridIndex,
                                data, ddata, PME_ORDER);
        scalar_t dpos[3] = {0, 0, 0};
        scalar_t dq = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = gridIndex[0] + ix;
            xindex -= (xindex >= gridx ? gridx : 0);
            scalar_t dx = data[ix][0], ddx = ddata[ix][0];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = gridIndex[1] + iy;
                yindex -= (yindex >= gridy ? gridy : 0);
                scalar_t dy = data[iy][1], ddy = ddata[iy][1];
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex[2] + iz;
                    zindex -= (zindex >= gridz ? gridz : 0);
                    scalar_t dz = data[iz][2], ddz = ddata[iz][2];
                    scalar_t g = grid[xindex][yindex][zindex];
                    dpos[0] += ddx * dy * dz * g;
                    dpos[1] += dx * ddy * dz * g;
                    dpos[2] += dx * dy * ddz * g;
                    dq += dx * dy * dz * g;
                }
            }
        }
        scalar_t scale = charge[atom] * sqrtCoulomb;
        posDeriv[atom][0] = scale * (dpos[0] * gridx * recipBoxVectors[0][0]);
        posDeriv[atom][1] = scale * (dpos[0] * gridx * recipBoxVectors[1][0]
                                   + dpos[1] * gridy * recipBoxVectors[1][1]);
        posDeriv[atom][2] = scale * (dpos[0] * gridx * recipBoxVectors[2][0]
                                   + dpos[1] * gridy * recipBoxVectors[2][1]
                                   + dpos[2] * gridz * recipBoxVectors[2][2]);
        chargeDeriv[atom] = dq * sqrtCoulomb;
    }
}

// ── autograd functions ─────────────────────────────────────────────────────

class PmeDirectFunctionCuda : public Function<PmeDirectFunctionCuda> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& positions, const Tensor& charges,
                          const Tensor& neighbors, const Tensor& deltas,
                          const Tensor& distances, const Tensor& exclusions,
                          const Scalar& alpha, const Scalar& coulomb) {
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = charges.size(0);
        int numPairs = neighbors.size(1);
        TensorOptions options = positions.options();
        Tensor posDeriv = torch::zeros({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::zeros({numAtoms}, options);
        int blockSize = 128;
        int numBlocks = std::max(1, std::min(getMaxBlocks(), (numPairs + blockSize - 1) / blockSize));
        Tensor energy = torch::zeros(numBlocks * blockSize, options);

        AT_DISPATCH_V2(positions.scalar_type(), "pme_direct::forward",
            AT_WRAP([&] {
                pme_direct_cuda_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                    get_accessor<scalar_t, 2>(positions),
                    get_accessor<scalar_t, 1>(charges),
                    get_accessor<int, 2>(neighbors),
                    get_accessor<scalar_t, 2>(deltas),
                    get_accessor<scalar_t, 1>(distances),
                    get_accessor<int, 2>(exclusions),
                    get_accessor<scalar_t, 2>(posDeriv),
                    get_accessor<scalar_t, 1>(chargeDeriv),
                    get_accessor<scalar_t, 1>(energy),
                    alpha.to<scalar_t>(), coulomb.to<scalar_t>());
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        ctx->save_for_backward({posDeriv, chargeDeriv});
        return torch::sum(energy);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor posDeriv = saved[0];
        Tensor chargeDeriv = saved[1];
        torch::Tensor ignore;
        return {posDeriv * grad_outputs[0], chargeDeriv * grad_outputs[0],
                ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

class PmeReciprocalFunctionCuda : public Function<PmeReciprocalFunctionCuda> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& positions, const Tensor& charges,
                          const Tensor& box_vectors,
                          const Scalar& gridx, const Scalar& gridy, const Scalar& gridz,
                          const Scalar& order,
                          const Scalar& alpha, const Scalar& coulomb,
                          const Tensor& xmoduli, const Tensor& ymoduli, const Tensor& zmoduli) {
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = positions.size(0);
        int pmeOrder = (int)order.toInt();
        int gridSize[3] = {(int)gridx.toInt(), (int)gridy.toInt(), (int)gridz.toInt()};
        TensorOptions options = positions.options();
        Tensor realGrid = torch::zeros({gridSize[0], gridSize[1], gridSize[2]}, options);
        int blockSize = 128;
        int numBlocks = std::max(1, std::min(getMaxBlocks(), (numAtoms + blockSize - 1) / blockSize));
        TORCH_CHECK(pmeOrder == 4 || pmeOrder == 5, "Only pmeOrder 4 or 5 is supported with CUDA");

        Tensor recipGrid;
        Tensor energy = torch::zeros(numBlocks * blockSize, options);

        AT_DISPATCH_V2(positions.scalar_type(), "pme_reciprocal::forward",
            AT_WRAP([&] {
                scalar_t sqrtCoulomb = std::sqrt(coulomb.to<scalar_t>());
                if (pmeOrder == 4) {
                    spreadCharge<scalar_t, 4><<<numBlocks, blockSize, 0, stream>>>(
                        get_accessor<scalar_t, 2>(positions),
                        get_accessor<scalar_t, 1>(charges),
                        get_accessor<scalar_t, 2>(box_vectors),
                        get_accessor<scalar_t, 3>(realGrid),
                        gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb);
                } else {
                    spreadCharge<scalar_t, 5><<<numBlocks, blockSize, 0, stream>>>(
                        get_accessor<scalar_t, 2>(positions),
                        get_accessor<scalar_t, 1>(charges),
                        get_accessor<scalar_t, 2>(box_vectors),
                        get_accessor<scalar_t, 3>(realGrid),
                        gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb);
                }
                recipGrid = torch::fft::rfftn(realGrid);
                scalar_t alpha_ = alpha.to<scalar_t>();
                reciprocalConvolution<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                    get_accessor<scalar_t, 2>(box_vectors),
                    get_accessor<c10::complex<scalar_t>, 3>(recipGrid),
                    gridSize[0], gridSize[1], gridSize[2],
                    get_accessor<scalar_t, 1>(xmoduli),
                    get_accessor<scalar_t, 1>(ymoduli),
                    get_accessor<scalar_t, 1>(zmoduli),
                    scalar_t(M_PI * M_PI) / (alpha_ * alpha_),
                    get_accessor<scalar_t, 1>(energy));
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        ctx->save_for_backward({positions, charges, box_vectors, xmoduli, ymoduli, zmoduli, recipGrid});
        ctx->saved_data["gridx"] = gridx;
        ctx->saved_data["gridy"] = gridy;
        ctx->saved_data["gridz"] = gridz;
        ctx->saved_data["order"] = order;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["coulomb"] = coulomb;
        return torch::sum(energy);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor positions = saved[0];
        Tensor charges = saved[1];
        Tensor box_vectors = saved[2];
        Tensor recipGrid = saved[6];
        int gridSize[3] = {(int)ctx->saved_data["gridx"].toInt(),
                           (int)ctx->saved_data["gridy"].toInt(),
                           (int)ctx->saved_data["gridz"].toInt()};
        int pmeOrder = (int)ctx->saved_data["order"].toInt();
        Scalar coulomb_s = ctx->saved_data["coulomb"].toScalar();
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = positions.size(0);
        TensorOptions options = positions.options();
        Tensor posDeriv = torch::empty({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::empty({numAtoms}, options);
        int blockSize = 128;
        int numBlocks = std::max(1, std::min(getMaxBlocks(), (numAtoms + blockSize - 1) / blockSize));
        TORCH_CHECK(pmeOrder == 4 || pmeOrder == 5, "Only pmeOrder 4 or 5 is supported with CUDA");

        int64_t targetGridSize[3] = {gridSize[0], gridSize[1], gridSize[2]};
        Tensor realGrid = torch::fft::irfftn(recipGrid, targetGridSize, c10::nullopt, "forward");

        AT_DISPATCH_V2(positions.scalar_type(), "pme_reciprocal::backward",
            AT_WRAP([&] {
                scalar_t sqrtCoulomb = std::sqrt(coulomb_s.to<scalar_t>());
                if (pmeOrder == 4) {
                    interpolateForce<scalar_t, 4><<<numBlocks, blockSize, 0, stream>>>(
                        get_accessor<scalar_t, 2>(positions),
                        get_accessor<scalar_t, 1>(charges),
                        get_accessor<scalar_t, 2>(box_vectors),
                        get_accessor<scalar_t, 3>(realGrid),
                        gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb,
                        get_accessor<scalar_t, 2>(posDeriv),
                        get_accessor<scalar_t, 1>(chargeDeriv));
                } else {
                    interpolateForce<scalar_t, 5><<<numBlocks, blockSize, 0, stream>>>(
                        get_accessor<scalar_t, 2>(positions),
                        get_accessor<scalar_t, 1>(charges),
                        get_accessor<scalar_t, 2>(box_vectors),
                        get_accessor<scalar_t, 3>(realGrid),
                        gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb,
                        get_accessor<scalar_t, 2>(posDeriv),
                        get_accessor<scalar_t, 1>(chargeDeriv));
                }
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        posDeriv *= grad_outputs[0];
        chargeDeriv *= grad_outputs[0];
        torch::Tensor ignore;
        return {posDeriv, chargeDeriv,
                ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

}  // namespace

static Tensor pme_direct_cuda(
    const Tensor& positions, const Tensor& charges, const Tensor& neighbors,
    const Tensor& deltas, const Tensor& distances, const Tensor& exclusions,
    const Scalar& alpha, const Scalar& coulomb
) {
    return PmeDirectFunctionCuda::apply(
        positions, charges, neighbors, deltas, distances, exclusions, alpha, coulomb);
}

static Tensor pme_reciprocal_cuda(
    const Tensor& positions, const Tensor& charges, const Tensor& box_vectors,
    const Scalar& gridx, const Scalar& gridy, const Scalar& gridz,
    const Scalar& order, const Scalar& alpha, const Scalar& coulomb,
    const Tensor& xmoduli, const Tensor& ymoduli, const Tensor& zmoduli
) {
    return PmeReciprocalFunctionCuda::apply(
        positions, charges, box_vectors, gridx, gridy, gridz, order, alpha, coulomb,
        xmoduli, ymoduli, zmoduli);
}

TORCH_LIBRARY_IMPL(molix, AutogradCUDA, m) {
    m.impl("pme_direct", pme_direct_cuda);
    m.impl("pme_reciprocal", pme_reciprocal_cuda);
}
