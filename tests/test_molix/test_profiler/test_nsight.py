"""NVIDIA command boundaries and artifact/error handling without a GPU."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from molix.profiler.nsight import NsightCompute, NsightSystems, NsightTool


class _ProcessTool(NsightTool):
    @property
    def report_name(self):
        return "test.report"

    def command(self, target, output):
        return [self.executable, *target, str(output / self.report_name)]


class TestNsightTool:
    @pytest.mark.parametrize("timeout", [0, float("nan"), float("inf")])
    def test_capture_requires_a_finite_budget(self, tmp_path, timeout):
        with pytest.raises(ValueError, match="finite positive"):
            _ProcessTool(sys.executable).run(["-c", "pass"], tmp_path / "invalid", timeout=timeout)
        assert not (tmp_path / "invalid").exists()

    def test_capture_preserves_arguments_and_report(self, tmp_path):
        tool = _ProcessTool(sys.executable)
        output = tmp_path / "directory with spaces"
        record = tool.run(
            ["-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('captured')"], output
        )
        assert record["status"] == "complete"
        assert Path(record["report"]).read_text() == "captured"
        assert json.loads((output / "capture.json").read_text())["returncode"] == 0
        with pytest.raises(FileExistsError):
            tool.run(["-c", "pass"], output)

    @pytest.mark.parametrize(
        "program,error",
        [
            ("raise SystemExit(7)", "exited 7"),
            ("pass", "no report"),
        ],
    )
    def test_failure_is_retained(self, tmp_path, program, error):
        output = tmp_path / "failed"
        with pytest.raises(RuntimeError, match=error):
            _ProcessTool(sys.executable).run(["-c", program], output)
        record = json.loads((output / "capture.json").read_text())
        assert record["status"] == "failed"
        assert error in record["error"]

    def test_timeout_stops_owned_process(self, tmp_path):
        output = tmp_path / "timeout"
        with pytest.raises(subprocess.TimeoutExpired):
            _ProcessTool(sys.executable).run(
                ["-c", "import time; time.sleep(30)"], output, timeout=0.05
            )
        assert json.loads((output / "capture.json").read_text())["status"] == "failed"

    def test_missing_executable_is_recorded(self, tmp_path):
        output = tmp_path / "missing"
        with pytest.raises(FileNotFoundError):
            _ProcessTool(str(tmp_path / "missing-tool")).run(["target"], output)
        assert json.loads((output / "capture.json").read_text())["tool"]["available"] is False


class TestNsightCompute:
    def test_raw_wide_csv_has_separate_units_and_kernel_rows(self):
        text = """"ID","Kernel Name","CC","gpu__time_duration.sum","dram__bytes_read.sum"
"","","","nsecond","byte"
"0","op<double, 128>","9.0","7000","1,024"
"""
        metrics = NsightCompute.parse_metrics(text)
        assert len(metrics) == 2
        assert metrics[0]["Metric Name"] == "gpu__time_duration.sum"
        assert metrics[0]["Metric Unit"] == "nsecond"
        assert metrics[1]["Metric Value"] == "1,024"
        assert metrics[1]["CC"] == "9.0"

    def test_parser_preserves_kernel_identity_units_and_quoted_names(self):
        text = """==PROF== Connected
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","op<float, 128>","dram__bytes_read.sum","byte","1,024"
"1","op<double, 128>","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","12.5"
"""
        metrics = NsightCompute.parse_metrics(text)
        assert metrics[0]["Kernel Name"] == "op<float, 128>"
        assert metrics[0]["Metric Value"] == "1,024"
        assert metrics[1]["Metric Unit"] == "%"

    def test_empty_or_permission_error_is_not_a_success(self):
        with pytest.raises(ValueError, match="no kernel metrics"):
            NsightCompute.parse_metrics("==ERROR== ERR_NVGPUCTRPERM")

    def test_command_is_bounded_and_does_not_set_clocks(self, tmp_path):
        command = NsightCompute(kernel="indexed_contract", launches=3).command(
            ["python", "a b.py", "--capture"], tmp_path
        )
        assert command[-3:] == ["python", "a b.py", "--capture"]
        assert command[command.index("--launch-count") + 1] == "3"
        assert command[command.index("--clock-control") + 1] == "none"
        assert command[command.index("--profile-from-start") + 1] == "off"


class TestNsightSystems:
    def test_stats_tables_keep_time_units_and_call_counts(self):
        text = """Processing timeline.sqlite
Time (%),Total Time (ns),Instances,Name
100,2500,3,"op<double, 128>"

Processing cuda_api_sum.py
Time (%),Total Time (ns),Num Calls,Name
100,3000,1,cudaDeviceSynchronize
"""
        tables = NsightSystems.parse_stats(text)
        assert tables["kernels"][0]["Name"] == "op<double, 128>"
        assert tables["kernels"][0]["Total Time (ns)"] == "2500"
        assert tables["cuda_api"][0]["Num Calls"] == "1"

    def test_capture_excludes_initialization_and_compile(self, tmp_path):
        command = NsightSystems().command(["python", "workload.py"], tmp_path)
        assert "--capture-range=cudaProfilerApi" in command
        assert "--cuda-graph-trace=node" in command

    def test_kernel_summary_per_iteration_without_templates_or_api_rows(self):
        tables = {
            "kernels": [
                {
                    "Name": "void ns::kernel<float, (int)1>(const float *, int)",
                    "Total Time (ns)": "3000000",
                    "Instances": "3",
                    "Avg (ns)": "1000000",
                },
                {
                    "Name": "void ns::kernel<double, (int)1>(const double *, int)",
                    "Total Time (ns)": "6000000",
                    "Instances": "3",
                    "Avg (ns)": "2000000",
                },
                {
                    "Name": "cudaDeviceSynchronize",
                    "Total Time (ns)": "9000000",
                    "Instances": "1",
                    "Avg (ns)": "9000000",
                },
                {
                    "Name": "cudaGraphLaunch_v10000",
                    "Total Time (ns)": "90000",
                    "Instances": "3",
                    "Avg (ns)": "30000",
                },
                {
                    "Name": "triton_poi_fused_add_7",
                    "Total Time (ns)": "300000",
                    "Instances": "3",
                    "Avg (ns)": "100000",
                },
            ],
            "cuda_api": [],
        }
        costs = NsightSystems.kernel_summary(tables, iterations=3)
        assert [c.name for c in costs] == ["ns::kernel", "ns::kernel", "triton_poi_fused_add_7"]
        assert [c.per_iteration_ms for c in costs] == pytest.approx([2.0, 1.0, 0.1])
        assert costs[0].mean_us == 2000.0 and costs[0].instances == 3
        assert NsightSystems.kernel_summary(tables, iterations=3, top=1) == costs[:1]
        table = NsightSystems.format_summary(costs)
        assert table.splitlines()[-1].strip().startswith("3.100 ms/iter")
        with pytest.raises(ValueError):
            NsightSystems.kernel_summary(tables, iterations=0)
