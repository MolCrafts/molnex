"""ptxas -v log parsing: pairing, spills, truncated and interleaved blocks."""

from molix.profiler.ptxas import KernelResources, demangle, format_resources, parse_ptxas

PRODUCT = "_ZN6molnex12equivariance14product_kernelIfLi1ELi3ELi128EEEvPKT_"
ADJOINT = "_ZN6molnex12equivariance14adjoint_kernelIdLi1ELi3ELi128ELb1ELi2EEEvPKT_"
LOG = f"""\
[ 50%] Building CUDA object CMakeFiles/x.dir/instance_1_3.cu.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '{PRODUCT}' for 'sm_90'
ptxas info    : Function properties for {PRODUCT}
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 122 registers, used 1 barriers, 9216 bytes smem
ptxas info    : Compiling entry function '{ADJOINT}' for 'sm_90'
ptxas info    : Function properties for {ADJOINT}
    0 bytes stack frame, 292 bytes spill stores, 308 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 34304 bytes smem
ptxas info    : Compiling entry function 'truncated' for 'sm_90'
"""


class TestParsePtxas:
    def test_entries_pair_their_registers_and_spills(self):
        kernels = parse_ptxas(LOG)
        assert [k.registers for k in kernels] == [122, 255]
        product, adjoint = kernels
        assert product == KernelResources(
            function=product.function,
            architecture="sm_90",
            registers=122,
            spill_stores=0,
            spill_loads=0,
            stack_frame=0,
            shared_memory=9216,
            barriers=1,
        )
        assert "product_kernelIfLi1ELi3ELi128E" in product.function
        assert not product.spills
        assert adjoint.spills and (adjoint.spill_stores, adjoint.spill_loads) == (292, 308)
        # The truncated last entry (no "Used" line) is dropped, not guessed.
        assert all(k.function != "truncated" for k in kernels)

    def test_properties_line_alone_still_pairs(self):
        text = (
            "ptxas info    : Function properties for _Z1kv\n"
            "    0 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads\n"
            "ptxas info    : Used 40 registers\n"
        )
        (kernel,) = parse_ptxas(text)
        assert kernel.function == "_Z1kv" and kernel.architecture == "?"
        assert (kernel.registers, kernel.spill_stores, kernel.shared_memory) == (40, 8, 0)

    def test_format_filters_by_pattern(self):
        table = format_resources(parse_ptxas(LOG), pattern="adjoint_kernelId")
        assert table.count("\n") == 0 and "255 regs" in table and "292/308" in table

    def test_demangle_without_tool_returns_input(self):
        names = ["_Z1kv"]
        assert demangle(names, executable="no-such-filter-binary") == names
