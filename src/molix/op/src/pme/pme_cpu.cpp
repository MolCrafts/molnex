// CPU implementation of PME direct/reciprocal ops.
//
// Dtype-generic via AT_DISPATCH_V2 (supports float/double). The reciprocal
// grid is allocated via torch::zeros (not std::vector + from_blob) so the
// backing storage is reference-counted and outlives the FFT call.

#include <ATen/Dispatch_v2.h>
#include <torch/extension.h>
#include <array>
#include <cmath>
#include <vector>

using namespace torch::autograd;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;

namespace {

// ── helpers ────────────────────────────────────────────────────────────────

template <typename T>
void invertBoxVectors(const torch::TensorAccessor<T, 2>& box, T recipBoxVectors[3][3]) {
    T determinant = box[0][0] * box[1][1] * box[2][2];
    T scale = T(1) / determinant;
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

template <typename T>
void computeSpline(
    int atom,
    const torch::TensorAccessor<T, 2>& pos,
    const torch::TensorAccessor<T, 2>& box,
    const T recipBoxVectors[3][3],
    const int gridSize[3],
    int gridIndex[3],
    std::vector<std::array<T, 3>>& data,
    std::vector<std::array<T, 3>>& ddata,
    int pmeOrder
) {
    T posInBox[3] = {pos[atom][0], pos[atom][1], pos[atom][2]};
    for (int i = 2; i >= 0; i--) {
        T scale = std::floor(posInBox[i] * recipBoxVectors[i][i]);
        for (int j = 0; j < 3; j++)
            posInBox[j] -= scale * box[i][j];
    }
    T t[3], dr[3];
    int ti[3];
    for (int i = 0; i < 3; i++) {
        t[i] = posInBox[0] * recipBoxVectors[0][i]
             + posInBox[1] * recipBoxVectors[1][i]
             + posInBox[2] * recipBoxVectors[2][i];
        t[i] = (t[i] - std::floor(t[i])) * gridSize[i];
        ti[i] = (int)t[i];
        dr[i] = t[i] - ti[i];
        gridIndex[i] = ti[i] % gridSize[i];
    }

    T scale = T(1) / (pmeOrder - 1);
    for (int i = 0; i < 3; i++) {
        data[pmeOrder - 1][i] = 0;
        data[1][i] = dr[i];
        data[0][i] = 1 - dr[i];
        for (int j = 3; j < pmeOrder; j++) {
            T div = T(1) / (j - 1);
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for (int k = 1; k < j - 1; k++)
                data[j - k - 1][i] = div * ((dr[i] + k) * data[j - k - 2][i]
                                          + (j - k - dr[i]) * data[j - k - 1][i]);
            data[0][i] = div * (1 - dr[i]) * data[0][i];
        }
        if (!ddata.empty()) {
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

// ── pme_direct ─────────────────────────────────────────────────────────────

template <typename scalar_t>
void pme_direct_kernel_cpu(
    const torch::TensorAccessor<scalar_t, 2>& pos,
    const torch::TensorAccessor<scalar_t, 1>& charge,
    const torch::TensorAccessor<int, 2>& pair,
    const torch::TensorAccessor<scalar_t, 2>& delta,
    const torch::TensorAccessor<scalar_t, 1>& r,
    const torch::TensorAccessor<int, 2>& excl,
    int numAtoms, int numPairs, int maxExclusions,
    scalar_t alpha, scalar_t coulomb,
    torch::TensorAccessor<scalar_t, 2> posDeriv_a,
    torch::TensorAccessor<scalar_t, 1> chargeDeriv_a,
    scalar_t& energy_out
) {
    scalar_t energy = 0;
    for (int i = 0; i < numPairs; i++) {
        int atom1 = pair[0][i];
        int atom2 = pair[1][i];
        bool include = (atom1 > -1);
        for (int j = 0; include && j < maxExclusions && excl[atom1][j] >= atom2; j++)
            if (excl[atom1][j] == atom2) include = false;
        if (!include) continue;

        scalar_t invR = 1 / r[i];
        scalar_t alphaR = alpha * r[i];
        scalar_t expAlphaRSqr = std::exp(-alphaR * alphaR);
        scalar_t erfcAlphaR = std::erfc(alphaR);
        scalar_t prefactor = coulomb * invR;
        scalar_t c1 = charge[atom1];
        scalar_t c2 = charge[atom2];
        energy += prefactor * erfcAlphaR * c1 * c2;
        chargeDeriv_a[atom1] += prefactor * erfcAlphaR * c2;
        chargeDeriv_a[atom2] += prefactor * erfcAlphaR * c1;
        scalar_t dEdR = prefactor * c1 * c2
                      * (erfcAlphaR + alphaR * expAlphaRSqr * M_2_SQRTPI)
                      * invR * invR;
        for (int j = 0; j < 3; j++) {
            posDeriv_a[atom1][j] -= dEdR * delta[i][j];
            posDeriv_a[atom2][j] += dEdR * delta[i][j];
        }
    }

    // Subtract excluded interactions added in reciprocal space.
    scalar_t dr[3];
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int i = 0; i < maxExclusions && excl[atom1][i] > atom1; i++) {
            int atom2 = excl[atom1][i];
            for (int j = 0; j < 3; j++) dr[j] = pos[atom1][j] - pos[atom2][j];
            scalar_t rr = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
            scalar_t invR = 1 / rr;
            scalar_t alphaR = alpha * rr;
            scalar_t expAlphaRSqr = std::exp(-alphaR * alphaR);
            scalar_t erfAlphaR = std::erf(alphaR);
            scalar_t prefactor = coulomb * invR;
            scalar_t c1 = charge[atom1];
            scalar_t c2 = charge[atom2];
            energy -= prefactor * erfAlphaR * c1 * c2;
            chargeDeriv_a[atom1] -= prefactor * erfAlphaR * c2;
            chargeDeriv_a[atom2] -= prefactor * erfAlphaR * c1;
            scalar_t dEdR = prefactor * c1 * c2
                          * (erfAlphaR - alphaR * expAlphaRSqr * M_2_SQRTPI)
                          * invR * invR;
            for (int j = 0; j < 3; j++) {
                posDeriv_a[atom1][j] += dEdR * dr[j];
                posDeriv_a[atom2][j] -= dEdR * dr[j];
            }
        }
    }
    energy_out = energy;
}

class PmeDirectFunctionCpu : public Function<PmeDirectFunctionCpu> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& positions,
                          const Tensor& charges,
                          const Tensor& neighbors,
                          const Tensor& deltas,
                          const Tensor& distances,
                          const Tensor& exclusions,
                          const Scalar& alpha_s,
                          const Scalar& coulomb_s) {
        int numAtoms = charges.size(0);
        int numPairs = neighbors.size(1);
        int maxExclusions = exclusions.size(1);
        TensorOptions options = positions.options();
        Tensor posDeriv = torch::zeros({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::zeros({numAtoms}, options);
        Tensor energy_t = torch::zeros({}, options);

        AT_DISPATCH_V2(positions.scalar_type(), "pme_direct::forward",
            AT_WRAP([&] {
                scalar_t energy_out = 0;
                pme_direct_kernel_cpu<scalar_t>(
                    positions.accessor<scalar_t, 2>(),
                    charges.accessor<scalar_t, 1>(),
                    neighbors.accessor<int, 2>(),
                    deltas.accessor<scalar_t, 2>(),
                    distances.accessor<scalar_t, 1>(),
                    exclusions.accessor<int, 2>(),
                    numAtoms, numPairs, maxExclusions,
                    alpha_s.to<scalar_t>(), coulomb_s.to<scalar_t>(),
                    posDeriv.accessor<scalar_t, 2>(),
                    chargeDeriv.accessor<scalar_t, 1>(),
                    energy_out);
                energy_t.fill_(energy_out);
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        ctx->save_for_backward({posDeriv, chargeDeriv});
        return energy_t;
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

// ── pme_reciprocal ─────────────────────────────────────────────────────────

template <typename scalar_t>
scalar_t pme_reciprocal_spread_and_convolve(
    const torch::TensorAccessor<scalar_t, 2>& pos,
    const torch::TensorAccessor<scalar_t, 1>& charge,
    const torch::TensorAccessor<scalar_t, 2>& box,
    const torch::TensorAccessor<scalar_t, 1>& xmod,
    const torch::TensorAccessor<scalar_t, 1>& ymod,
    const torch::TensorAccessor<scalar_t, 1>& zmod,
    int numAtoms, int pmeOrder,
    const int gridSize[3],
    const scalar_t recipBoxVectors[3][3],
    scalar_t sqrtCoulomb, scalar_t alpha,
    Tensor& recipGrid_out   // output: complex grid post-FFT
) {
    // Spread charge onto real grid (allocated as torch::zeros for safe lifetime).
    Tensor realGrid_t = torch::zeros(
        {gridSize[0], gridSize[1], gridSize[2]},
        torch::TensorOptions().dtype(c10::CppTypeToScalarType<scalar_t>::value));
    auto grid = realGrid_t.accessor<scalar_t, 3>();

    for (int atom = 0; atom < numAtoms; atom++) {
        int gridIndex[3];
        std::vector<std::array<scalar_t, 3>> data(pmeOrder), ddata;
        computeSpline<scalar_t>(atom, pos, box, recipBoxVectors, gridSize, gridIndex, data, ddata, pmeOrder);
        for (int ix = 0; ix < pmeOrder; ix++) {
            int xindex = (gridIndex[0] + ix) % gridSize[0];
            scalar_t dx = charge[atom] * sqrtCoulomb * data[ix][0];
            for (int iy = 0; iy < pmeOrder; iy++) {
                int yindex = (gridIndex[1] + iy) % gridSize[1];
                scalar_t dxdy = dx * data[iy][1];
                for (int iz = 0; iz < pmeOrder; iz++) {
                    int zindex = (gridIndex[2] + iz) % gridSize[2];
                    grid[xindex][yindex][zindex] += dxdy * data[iz][2];
                }
            }
        }
    }

    // FFT
    recipGrid_out = torch::fft::rfftn(realGrid_t);
    auto recip = recipGrid_out.accessor<c10::complex<scalar_t>, 3>();

    // Convolution + energy accumulation
    scalar_t energy = 0;
    int zsize = gridSize[2] / 2 + 1;
    int yzsize = gridSize[1] * zsize;
    (void)yzsize;  // derivable; retained in CUDA path; unused here
    scalar_t scaleFactor = (scalar_t)(M_PI * box[0][0] * box[1][1] * box[2][2]);
    scalar_t recipExpFactor = (scalar_t)(M_PI * M_PI / (alpha * alpha));
    for (int kx = 0; kx < gridSize[0]; kx++) {
        int mx = (kx < (gridSize[0] + 1) / 2) ? kx : kx - gridSize[0];
        scalar_t mhx = mx * recipBoxVectors[0][0];
        scalar_t bx = scaleFactor * xmod[kx];
        for (int ky = 0; ky < gridSize[1]; ky++) {
            int my = (ky < (gridSize[1] + 1) / 2) ? ky : ky - gridSize[1];
            scalar_t mhy = mx * recipBoxVectors[1][0] + my * recipBoxVectors[1][1];
            scalar_t mhx2y2 = mhx * mhx + mhy * mhy;
            scalar_t bxby = bx * ymod[ky];
            for (int kz = 0; kz < zsize; kz++) {
                int mz = (kz < (gridSize[2] + 1) / 2) ? kz : kz - gridSize[2];
                scalar_t mhz = mx * recipBoxVectors[2][0]
                             + my * recipBoxVectors[2][1]
                             + mz * recipBoxVectors[2][2];
                scalar_t bz = zmod[kz];
                scalar_t m2 = mhx2y2 + mhz * mhz;
                scalar_t denom = m2 * bxby * bz;
                int index = kx * gridSize[1] * zsize + ky * zsize + kz;
                scalar_t eterm = (index == 0 ? scalar_t(0) : std::exp(-recipExpFactor * m2) / denom);
                scalar_t scale = (kz > 0 && kz <= (gridSize[2] - 1) / 2 ? scalar_t(2) : scalar_t(1));
                c10::complex<scalar_t>& g = recip[kx][ky][kz];
                energy += scale * eterm * (g.real() * g.real() + g.imag() * g.imag());
                g *= eterm;
            }
        }
    }
    return energy;
}

template <typename scalar_t>
void pme_reciprocal_backward_kernel_cpu(
    const torch::TensorAccessor<scalar_t, 2>& pos,
    const torch::TensorAccessor<scalar_t, 1>& charge,
    const torch::TensorAccessor<scalar_t, 2>& box,
    const torch::TensorAccessor<scalar_t, 3>& grid,
    int numAtoms, int pmeOrder,
    const int gridSize[3],
    const scalar_t recipBoxVectors[3][3],
    scalar_t sqrtCoulomb,
    torch::TensorAccessor<scalar_t, 2> posDeriv_a,
    torch::TensorAccessor<scalar_t, 1> chargeDeriv_a
) {
    for (int atom = 0; atom < numAtoms; atom++) {
        int gridIndex[3];
        std::vector<std::array<scalar_t, 3>> data(pmeOrder), ddata(pmeOrder);
        computeSpline<scalar_t>(atom, pos, box, recipBoxVectors, gridSize, gridIndex, data, ddata, pmeOrder);

        scalar_t dpos[3] = {0, 0, 0};
        scalar_t dq = 0;
        for (int ix = 0; ix < pmeOrder; ix++) {
            int xindex = (gridIndex[0] + ix) % gridSize[0];
            scalar_t dx = data[ix][0], ddx = ddata[ix][0];
            for (int iy = 0; iy < pmeOrder; iy++) {
                int yindex = (gridIndex[1] + iy) % gridSize[1];
                scalar_t dy = data[iy][1], ddy = ddata[iy][1];
                for (int iz = 0; iz < pmeOrder; iz++) {
                    int zindex = (gridIndex[2] + iz) % gridSize[2];
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
        posDeriv_a[atom][0] = scale * (dpos[0] * gridSize[0] * recipBoxVectors[0][0]);
        posDeriv_a[atom][1] = scale * (dpos[0] * gridSize[0] * recipBoxVectors[1][0]
                                     + dpos[1] * gridSize[1] * recipBoxVectors[1][1]);
        posDeriv_a[atom][2] = scale * (dpos[0] * gridSize[0] * recipBoxVectors[2][0]
                                     + dpos[1] * gridSize[1] * recipBoxVectors[2][1]
                                     + dpos[2] * gridSize[2] * recipBoxVectors[2][2]);
        chargeDeriv_a[atom] = dq * sqrtCoulomb;
    }
}

class PmeReciprocalFunctionCpu : public Function<PmeReciprocalFunctionCpu> {
public:
    static Tensor forward(AutogradContext* ctx,
                          const Tensor& positions,
                          const Tensor& charges,
                          const Tensor& box_vectors,
                          const Scalar& gridx, const Scalar& gridy, const Scalar& gridz,
                          const Scalar& order,
                          const Scalar& alpha, const Scalar& coulomb,
                          const Tensor& xmoduli, const Tensor& ymoduli, const Tensor& zmoduli) {
        int numAtoms = positions.size(0);
        int pmeOrder = (int)order.toInt();
        int gridSize[3] = {(int)gridx.toInt(), (int)gridy.toInt(), (int)gridz.toInt()};
        TensorOptions options = positions.options();
        Tensor energy_t = torch::zeros({}, options);
        Tensor recipGrid;

        AT_DISPATCH_V2(positions.scalar_type(), "pme_reciprocal::forward",
            AT_WRAP([&] {
                scalar_t sqrtCoulomb = std::sqrt(coulomb.to<scalar_t>());
                scalar_t alpha_ = alpha.to<scalar_t>();
                scalar_t recipBoxVectors[3][3];
                invertBoxVectors<scalar_t>(box_vectors.accessor<scalar_t, 2>(), recipBoxVectors);
                scalar_t energy = pme_reciprocal_spread_and_convolve<scalar_t>(
                    positions.accessor<scalar_t, 2>(),
                    charges.accessor<scalar_t, 1>(),
                    box_vectors.accessor<scalar_t, 2>(),
                    xmoduli.accessor<scalar_t, 1>(),
                    ymoduli.accessor<scalar_t, 1>(),
                    zmoduli.accessor<scalar_t, 1>(),
                    numAtoms, pmeOrder, gridSize, recipBoxVectors,
                    sqrtCoulomb, alpha_, recipGrid);
                energy_t.fill_(scalar_t(0.5) * energy);
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        ctx->save_for_backward({positions, charges, box_vectors, xmoduli, ymoduli, zmoduli, recipGrid});
        ctx->saved_data["gridx"] = gridx;
        ctx->saved_data["gridy"] = gridy;
        ctx->saved_data["gridz"] = gridz;
        ctx->saved_data["order"] = order;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["coulomb"] = coulomb;
        return energy_t;
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
        int numAtoms = positions.size(0);
        TensorOptions options = positions.options();
        Tensor posDeriv = torch::empty({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::empty({numAtoms}, options);

        int64_t targetGridSize[3] = {gridSize[0], gridSize[1], gridSize[2]};
        Tensor realGrid = torch::fft::irfftn(recipGrid, targetGridSize, c10::nullopt, "forward");

        AT_DISPATCH_V2(positions.scalar_type(), "pme_reciprocal::backward",
            AT_WRAP([&] {
                scalar_t sqrtCoulomb = std::sqrt(coulomb_s.to<scalar_t>());
                scalar_t recipBoxVectors[3][3];
                invertBoxVectors<scalar_t>(box_vectors.accessor<scalar_t, 2>(), recipBoxVectors);
                pme_reciprocal_backward_kernel_cpu<scalar_t>(
                    positions.accessor<scalar_t, 2>(),
                    charges.accessor<scalar_t, 1>(),
                    box_vectors.accessor<scalar_t, 2>(),
                    realGrid.accessor<scalar_t, 3>(),
                    numAtoms, pmeOrder, gridSize, recipBoxVectors, sqrtCoulomb,
                    posDeriv.accessor<scalar_t, 2>(),
                    chargeDeriv.accessor<scalar_t, 1>());
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        torch::Tensor ignore;
        return {posDeriv * grad_outputs[0], chargeDeriv * grad_outputs[0],
                ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

}  // namespace

static Tensor pme_direct_cpu(
    const Tensor& positions, const Tensor& charges, const Tensor& neighbors,
    const Tensor& deltas, const Tensor& distances, const Tensor& exclusions,
    const Scalar& alpha, const Scalar& coulomb
) {
    return PmeDirectFunctionCpu::apply(
        positions, charges, neighbors, deltas, distances, exclusions, alpha, coulomb);
}

static Tensor pme_reciprocal_cpu(
    const Tensor& positions, const Tensor& charges, const Tensor& box_vectors,
    const Scalar& gridx, const Scalar& gridy, const Scalar& gridz,
    const Scalar& order, const Scalar& alpha, const Scalar& coulomb,
    const Tensor& xmoduli, const Tensor& ymoduli, const Tensor& zmoduli
) {
    return PmeReciprocalFunctionCpu::apply(
        positions, charges, box_vectors, gridx, gridy, gridz, order, alpha, coulomb,
        xmoduli, ymoduli, zmoduli);
}

TORCH_LIBRARY_IMPL(molix, AutogradCPU, m) {
    m.impl("pme_direct", pme_direct_cpu);
    m.impl("pme_reciprocal", pme_reciprocal_cpu);
}
