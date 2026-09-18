"""Tests for `kspace_filter` classes"""

import pytest
import torch

from molpot.potentials.elec.lib import (
    KSpaceFilter,
    KSpaceKernel,
    MeshInterpolator,
    generate_kvectors_for_mesh,
)


class TestKernel:
    class DemoKernel(KSpaceKernel):
        def __init__(self, param: float):
            super().__init__()
            self.param = param

        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.exp(-k_sq / self.param)

    class NoopKernel(KSpaceKernel):
        def __init__(self):
            super().__init__()

        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(k_sq)

    def test_kernel_subclassing(self):
        # check that one can define and use a kernel
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)

        my_krn.kernel_from_k_sq(k_sq)

    def test_kernel_compile(self):
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)
        filter = my_krn.kernel_from_k_sq(k_sq)

        compiled_kernel = torch.compile(my_krn.kernel_from_k_sq, backend="eager")
        compiled_filter = compiled_kernel(k_sq)

        assert torch.allclose(filter, compiled_filter)


class TestFilter:
    cell1 = torch.randn((3, 3))
    cell2 = torch.randn((3, 3))
    ns1 = torch.tensor([3, 4, 5])
    ns2 = torch.tensor([4, 2, 1])

    mykernel = TestKernel.DemoKernel(1.0)
    myfilter1 = KSpaceFilter(cell1, ns1, mykernel)
    myfilter2 = KSpaceFilter(cell2, ns2, mykernel)
    mymesh1 = MeshInterpolator(cell1, ns1, 3, method="P3M")
    mymesh2 = MeshInterpolator(cell2, ns2, 3, method="Lagrange")
    points = torch.tensor([[1.0, 2, 3], [0, 1, 1]])
    weights = torch.tensor([[-0.1], [0.4]])

    mykernel_noop = TestKernel.NoopKernel()
    myfilter_noop = KSpaceFilter(cell1, ns1, mykernel_noop)

    def test_meshes_consistent_size(self):
        # make sure we get conistent mesh sizes
        stencil = self.mymesh1.compute_weights(self.points, self.cell1)
        mesh = self.mymesh1.points_to_mesh(self.weights, stencil)
        # nb - the third value is different because of the real-valued FT
        assert mesh.shape[1:3] == self.myfilter1.kfilter(self.cell1).shape[:-1]

    def test_meshes_inconsistent_size(self):
        # make sure we get consistent mesh sizes
        stencil = self.mymesh1.compute_weights(self.points, self.cell1)
        mesh = self.mymesh1.points_to_mesh(self.weights, stencil)
        match = "The real-space mesh is inconsistent with the k-space grid."
        with pytest.raises(ValueError, match=match):
            self.myfilter2.forward(mesh, self.cell2)

    def test_kernel_noop(self):
        # make sure that a filter of ones recovers the initial mesh
        stencil = self.mymesh1.compute_weights(self.points, self.cell1)
        mesh = self.mymesh1.points_to_mesh(self.weights, stencil)
        mesh_transformed = self.myfilter_noop.forward(mesh, self.cell1)

        assert torch.allclose(mesh, mesh_transformed, atol=1e-6, rtol=0)

    def test_filter_linear(self):
        # checks that the filter (as well as the mesh interpolator) are linear
        stencil = self.mymesh1.compute_weights(self.points, self.cell1)
        mesh1 = self.mymesh1.points_to_mesh(self.weights, stencil)

        mesh2 = torch.exp(mesh1)

        tmesh1 = self.myfilter1.forward(mesh1, self.cell1)
        tmesh2 = self.myfilter1.forward(mesh2, self.cell1)
        tmesh12 = self.myfilter1.forward(mesh1 + 0.3 * mesh2, self.cell1)

        assert torch.allclose(tmesh12, tmesh1 + 0.3 * tmesh2, atol=1e-6, rtol=1e-5)


def test_update():
    kernel_filter = KSpaceFilter(
        cell=2 * torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    kernel_filter.update(ns_mesh=torch.tensor([3, 3, 3]))

    assert torch.all(kernel_filter.ns_mesh == torch.tensor([3, 3, 3]))


@pytest.mark.parametrize("cell", [2 * torch.eye(3), torch.randn(3, 3)])
def test_kfilter_follows_the_cell_and_the_kernel_it_is_called_with(cell):
    """Nothing is cached: the filter is the live kernel on the k-grid of the given cell."""
    ns_mesh = torch.tensor([2, 2, 2])
    kernel = TestKernel.DemoKernel(1.0)
    kernel_filter = KSpaceFilter(cell=torch.eye(3), ns_mesh=ns_mesh, kernel=kernel)

    # A kernel parameter changed after construction is seen on the next call.
    kernel.param = 2.0

    kvectors = generate_kvectors_for_mesh(ns=ns_mesh, cell=cell)
    k_sq = torch.linalg.norm(kvectors, dim=3) ** 2
    torch.testing.assert_close(kernel_filter.kfilter(cell), kernel.kernel_from_k_sq(k_sq))


def test_update_ns_wrong_shape():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "shape \\[2\\] of `ns_mesh` has to be \\(3,\\)"
    with pytest.raises(ValueError, match=match):
        kernel_filter.update(ns_mesh=torch.tensor([2, 2]))


def test_kfilter_cell_wrong_shape():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        kernel_filter.kfilter(torch.tensor([[1.0, 0, 0], [0, 1, 0]]))


def test_kfilter_devices_ns_cell():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "`cell` and `ns_mesh` are on different devices, got meta and cpu"
    with pytest.raises(ValueError, match=match):
        kernel_filter.kfilter(torch.eye(3, device="meta"))


def test_fft_modes():
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], device="cpu")
    match = "Invalid option 'faster' for the `fft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), fft_norm="faster")
    match = "Invalid option 'faster' for the `ifft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), ifft_norm="faster")
