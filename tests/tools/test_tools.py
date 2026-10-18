"""Tests for the repository tools: Roost boundaries, gpu-lease and the backlog parser.

python3 -m pytest tests/tools
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(TOOLS / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


boundaries = load("check_roost_boundaries", "check-roost-boundaries.py")
gpu_lease = load("gpu_lease", "gpu-lease")
backlog = load("create_backlog_issues", "create_backlog_issues.py")


def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "Roost.slnx").write_text('<Solution><Project Path="src/A/A.csproj" /></Solution>')
    return tmp_path


def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


# check-roost-boundaries.py


def test_clean_repository_passes(tmp_path):
    root = git_repo(tmp_path)
    write(root / "src/Starlings.Roost.Ops/CpuOps.cs", "namespace Starlings.Roost;\n")
    assert boundaries.main(["--root", str(root), "--no-closure"]) == 0


def test_python_under_src_fails(tmp_path, capsys):
    root = git_repo(tmp_path)
    write(root / "src/Starlings.Roost.Ops/helper.py", "print()\n")
    write(root / "src/Starlings.Roost.Worker/Interop.cs", 'const string Lib = "libtorch_cuda";\n')
    write(root / "tools/reference.py", "import torch\n")  # allowed outside src/
    assert boundaries.main(["--root", str(root), "--no-closure"]) == 1
    out = capsys.readouterr().out
    assert "helper.py: Python source under src/" in out
    assert "Interop.cs:1: references 'libtorch'" in out
    assert "tools/reference.py" not in out


def test_native_binary_fails_by_extension_and_magic(tmp_path, capsys):
    root = git_repo(tmp_path)
    write(root / "kernels/libroost_kernels.so", b"\x7fELF....")
    write(root / "kernels/blob.bin", b"\x7fELF....")
    assert boundaries.main(["--root", str(root), "--no-closure"]) == 1
    out = capsys.readouterr().out
    assert "kernels/libroost_kernels.so: native binary" in out
    assert "kernels/blob.bin: native binary tracked in Git (magic 7f454c46)" in out


def test_cuda_project_in_solution_fails(tmp_path, capsys):
    root = git_repo(tmp_path)
    (root / "Roost.slnx").write_text(
        '<Solution><Project Path="tests/Starlings.Roost.Worker.Tests.Cuda/Starlings.Roost.Worker.Tests.Cuda.csproj" />'
        '<Project Path="src/Starlings.Roost.Cuda/Starlings.Roost.Cuda.csproj" />'
        '<Project Path="src/Starlings.Roost.CudaLike/Starlings.Roost.CudaLike.csproj" /></Solution>'
    )
    assert boundaries.main(["--root", str(root), "--no-closure"]) == 1
    out = capsys.readouterr().out
    assert out.count("solution:") == 2
    assert "CudaLike" not in out


def assets(names: list[str]) -> str:
    return json.dumps({"libraries": {f"{n}/1.0.0": {} for n in names}})


def test_onnx_runtime_in_engine_closure_fails(tmp_path, capsys):
    root = git_repo(tmp_path)
    write(root / "src/Starlings.Roost.Models.Qwen3/Starlings.Roost.Models.Qwen3.csproj", "<Project />")
    write(
        root / "src/Starlings.Roost.Models.Qwen3/obj/project.assets.json",
        assets(["Starlings.Tokenizers", "Microsoft.ML.OnnxRuntimeGenAI.Managed"]),
    )
    write(root / "src/Starlings.Roost.Server/Starlings.Roost.Server.csproj", "<Project />")
    write(root / "src/Starlings.Roost.Server/obj/project.assets.json", assets(["Microsoft.ML.OnnxRuntime"]))
    assert boundaries.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "Models.Qwen3.csproj: engine package reaches Microsoft.ML.OnnxRuntimeGenAI.Managed/1.0.0" in out
    assert "Server" not in out  # exempt (R-052)


def test_unrestored_engine_project_fails(tmp_path, capsys):
    root = git_repo(tmp_path)
    write(root / "src/Starlings.Generation.Roost/Starlings.Generation.Roost.csproj", "<Project />")
    assert boundaries.main(["--root", str(root)]) == 1
    assert "not restored" in capsys.readouterr().out


# gpu-lease


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    def install(free_mib: int) -> None:
        script = tmp_path / "nvidia-smi"
        script.write_text(
            "#!/bin/sh\n"
            f"echo '0, GPU-aaaa, NVIDIA GeForce RTX 3090, 24576, {free_mib}'\n"
            "echo '1, GPU-bbbb, NVIDIA GeForce RTX 4060 Ti, 16380, 800'\n"
        )
        script.chmod(0o755)

    monkeypatch.setenv("ROOST_NVIDIA_SMI", str(tmp_path / "nvidia-smi"))
    monkeypatch.setenv("ROOST_GPU_LEASE_DIR", str(tmp_path / "state"))
    return install


def ledger(tmp_path: Path) -> list[dict]:
    path = tmp_path / "state" / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_lease_runs_command_on_the_device_and_records_it(tmp_path, fake_smi):
    fake_smi(free_mib=20000)
    out = tmp_path / "env.txt"
    code = gpu_lease.main(
        [
            "run",
            "--device",
            "3090",
            "--need-mib",
            "8000",
            "--stream",
            "F",
            "--",
            sys.executable,
            "-c",
            f"import os; open({str(out)!r}, 'w').write(os.environ['CUDA_VISIBLE_DEVICES'])",
        ]
    )
    assert code == 0
    assert out.read_text() == "GPU-aaaa"
    [entry] = ledger(tmp_path)
    assert entry["device"] == "NVIDIA GeForce RTX 3090"
    assert entry["stream"] == "F"
    assert entry["exit_code"] == 0


def test_lease_refuses_when_memory_is_short(tmp_path, fake_smi):
    fake_smi(free_mib=20000)
    code = gpu_lease.main(["run", "--device", "4060", "--need-mib", "8000", "--stream", "C", "--", "true"])
    assert code == gpu_lease.EX_TEMPFAIL
    assert ledger(tmp_path) == []


def test_lease_times_out_while_another_holds_the_device(tmp_path, fake_smi):
    fake_smi(free_mib=20000)
    holder = subprocess.Popen(
        [
            sys.executable,
            str(TOOLS / "gpu-lease"),
            "run",
            "--device",
            "0",
            "--need-mib",
            "1",
            "--stream",
            "D",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
        env=os.environ.copy(),
    )
    try:
        lock = tmp_path / "state" / "GPU-aaaa.lock"
        for _ in range(100):
            if lock.exists() and lock.read_text():
                break
            subprocess.run(["sleep", "0.05"], check=True)
        code = gpu_lease.main(
            ["run", "--device", "0", "--need-mib", "1", "--stream", "D", "--wait", "0.2", "--", "true"]
        )
        assert code == gpu_lease.EX_TEMPFAIL
    finally:
        holder.kill()
        holder.wait()


def test_list_ignores_holder_text_left_by_a_killed_lease(tmp_path, fake_smi, capsys):
    fake_smi(free_mib=20000)
    state = tmp_path / "state"
    state.mkdir()
    (state / "GPU-aaaa.lock").write_text('{"lease": "stale"}')
    assert gpu_lease.main(["list"]) == 0
    assert "stale" not in capsys.readouterr().out


def test_ambiguous_device_is_a_usage_error(fake_smi):
    fake_smi(free_mib=20000)
    with pytest.raises(SystemExit, match="matches 2"):
        gpu_lease.main(["run", "--device", "RTX", "--need-mib", "1", "--stream", "A", "--", "true"])


# create_backlog_issues.py


def test_backlog_parses_every_item_with_labels():
    items = backlog.parse_backlog(backlog.PLAN.read_text(encoding="utf-8"))
    assert len(items) == backlog.EXPECTED
    by_id = {item["id"]: item for item in items}
    assert backlog.labels_for(by_id["R-060"]) == ["backlog", "stream:A", "phase:M0", "phase:M2"]
    assert "starling-pr" in backlog.labels_for(by_id["R-009"])
    assert "starling-pr" in backlog.labels_for(by_id["R-001"])
    assert "starling-pr" not in backlog.labels_for(by_id["R-003"])
