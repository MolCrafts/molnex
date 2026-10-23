"""CLI argument forwarding must not reinterpret a workload's flags."""

from typer.testing import CliRunner

from molix.cli.profile import app
from molix.profiler.nsight import NsightSystems


class TestProfileCLI:
    def test_workload_options_after_separator(self, monkeypatch, tmp_path):
        received = []

        def run(self, target, output, *, timeout):
            received.append(target)
            return {"report": str(output / "timeline.nsys-rep")}

        monkeypatch.setattr(NsightSystems, "run", run)
        monkeypatch.setattr(NsightSystems, "stats", lambda *args: None)
        result = CliRunner().invoke(
            app,
            [
                "nsys",
                "--output",
                str(tmp_path),
                "--",
                "python",
                "with spaces.py",
                "--output",
                "workload-output",
                "--capture",
            ],
        )
        assert result.exit_code == 0, result.output
        assert received == [
            ["python", "with spaces.py", "--output", "workload-output", "--capture"]
        ]

    def test_summarize_reads_stats_json_and_csv(self, tmp_path):
        import json

        tables = {
            "kernels": [
                {
                    "Name": "void k<float>(float *)",
                    "Total Time (ns)": "2000000",
                    "Instances": "2",
                    "Avg (ns)": "1000000",
                },
                {
                    "Name": "cudaDeviceSynchronize",
                    "Total Time (ns)": "5000000",
                    "Instances": "1",
                    "Avg (ns)": "5000000",
                },
            ],
            "cuda_api": [],
        }
        stats = tmp_path / "stats.json"
        stats.write_text(json.dumps(tables))
        result = CliRunner().invoke(app, ["summarize", str(stats), "--iterations", "2"])
        assert result.exit_code == 0, result.output
        assert "1.000 ms/iter" in result.output and "cudaDeviceSynchronize" not in result.output
        csv_stats = tmp_path / "stats.csv"
        csv_stats.write_text(
            "Processing report\n"
            "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),"
            "StdDev (ns),Name\n"
            '100.0,2000000,2,1000000,1000000,1000000,1000000,0,"void k<float>(float *)"\n'
        )
        result = CliRunner().invoke(app, ["summarize", str(csv_stats), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)[0]["name"] == "k"

    def test_ptxas_reports_registers_and_spills(self, tmp_path):
        log = tmp_path / "build.log"
        log.write_text(
            "ptxas info    : Compiling entry function '_Z1kv' for 'sm_90'\n"
            "ptxas info    : Function properties for _Z1kv\n"
            "    0 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads\n"
            "ptxas info    : Used 200 registers, used 1 barriers, 1024 bytes smem\n"
        )
        result = CliRunner().invoke(app, ["ptxas", str(log), "--pattern", "_Z1kv"])
        assert result.exit_code == 0, result.output
        assert "200 regs" in result.output and "16/8" in result.output
