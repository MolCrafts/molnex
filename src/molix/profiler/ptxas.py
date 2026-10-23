"""Per-kernel register, spill and shared-memory usage from a ``ptxas -v`` build log.

Register pressure decided every round of the native channelwise optimisation
(2026-10-23): a kernel pushed to 255 registers spills, a launch bound that
forces fewer spills more, and both show up only here, not in a timeline. The
parser pairs each ``Compiling entry function`` line with the ``Function
properties`` / ``Used N registers`` lines ptxas prints for it. Build with
``-Xptxas=-v`` (``-DMOLNEX_EQUIVARIANCE_PTXAS_VERBOSE=ON`` for the equivariance
library); a parallel ``make`` may interleave the blocks of different
translation units, ``ninja`` prints each job's output whole.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PROPERTIES = re.compile(r"Function properties for (\S+)")
_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?")


@dataclass(frozen=True)
class KernelResources:
    """What ptxas allocated for one kernel entry on one architecture."""

    function: str  #: mangled entry name (``demangle`` for a readable one)
    architecture: str  #: e.g. ``sm_90``
    registers: int
    spill_stores: int  #: bytes
    spill_loads: int  #: bytes
    stack_frame: int  #: bytes
    shared_memory: int  #: static bytes
    barriers: int

    @property
    def spills(self) -> bool:
        return self.spill_stores > 0 or self.spill_loads > 0


def parse_ptxas(text: str) -> list[KernelResources]:
    """Every kernel entry of a ``ptxas -v`` log, in order of appearance.

    An entry is complete once its ``Used … registers`` line arrives; an entry
    without one (a truncated log) is dropped.
    """
    kernels: list[KernelResources] = []
    pending: dict[str, int | str] = {}
    for line in text.splitlines():
        if match := _ENTRY.search(line):
            pending = {"function": match.group(1), "architecture": match.group(2)}
        elif match := _PROPERTIES.search(line):
            if pending.get("function") != match.group(1):
                pending = {"function": match.group(1), "architecture": "?"}
        elif (match := _STACK.search(line)) and pending:
            pending["stack_frame"], pending["spill_stores"], pending["spill_loads"] = (
                int(x) for x in match.groups()
            )
        elif (match := _USED.search(line)) and pending:
            registers, barriers, smem = match.groups()
            kernels.append(
                KernelResources(
                    function=str(pending["function"]),
                    architecture=str(pending["architecture"]),
                    registers=int(registers),
                    spill_stores=int(pending.get("spill_stores", 0)),
                    spill_loads=int(pending.get("spill_loads", 0)),
                    stack_frame=int(pending.get("stack_frame", 0)),
                    shared_memory=int(smem or 0),
                    barriers=int(barriers or 0),
                )
            )
            pending = {}
    return kernels


def demangle(names: list[str], *, executable: str = "c++filt") -> list[str]:
    """Readable names through ``c++filt`` when it is installed; the input otherwise."""
    tool = shutil.which(executable)
    if tool is None or not names:
        return list(names)
    result = subprocess.run(
        [tool], input="\n".join(names) + "\n", capture_output=True, text=True, check=False
    )
    lines = result.stdout.splitlines()
    return lines if result.returncode == 0 and len(lines) == len(names) else list(names)


def format_resources(kernels: list[KernelResources], *, pattern: str | None = None) -> str:
    """One line per kernel matching ``pattern`` (a regex on the entry name)."""
    selected = [k for k in kernels if pattern is None or re.search(pattern, k.function)]
    rows = [
        f"{k.registers:4d} regs  {k.spill_stores:5d}/{k.spill_loads:<5d} B spill  "
        f"{k.shared_memory:6d} B smem  {k.function}"
        for k in selected
    ]
    return "\n".join(rows)


def resources_as_dicts(kernels: list[KernelResources]) -> list[dict]:
    return [asdict(k) for k in kernels]
