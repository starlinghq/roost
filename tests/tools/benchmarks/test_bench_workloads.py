"""Workload files (benchmarks/workloads) and their deterministic expansion."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from roost_bench import workloads
from roost_bench.workloads import Dist, WorkloadError

ROOT = Path(__file__).resolve().parents[3]
FILES = sorted((ROOT / "benchmarks" / "workloads").glob("*.toml"))


def test_every_section_14_3_workload_has_a_file():
    ids = {workloads.load(p).id.split("-")[0] for p in FILES}
    assert ids == {"w1", "w2", "w3", "w4", "w5", "w6", "w7"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_committed_workloads_parse_and_expand(path: Path):
    w = workloads.load(path)
    assert w.status in workloads.STATUSES
    assert w.status != "supported"  # nothing has been measured on a GPU
    assert w.id == path.stem
    sessions = w.sessions(seed=0)
    assert len(sessions) == w.requests
    assert all(s.steps for s in sessions)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_expansion_is_deterministic_per_seed(path: Path):
    w = workloads.load(path)
    a, b, c = w.sessions(seed=1), w.sessions(seed=1), w.sessions(seed=2)
    assert a == b
    assert a != c


def test_slo_file_is_marked_proposed():
    raw = tomllib.loads((ROOT / "benchmarks" / "slo.toml").read_text())
    assert raw["status"] == "proposed"
    assert raw["decision"] == 25
    assert (raw["ttft_p99_ms"], raw["itl_p99_ms"]) == (1000, 50)


def test_w3_sessions_carry_history_over_shared_prefixes():
    w = workloads.load(ROOT / "benchmarks/workloads/w3-agentic-multiturn.toml")
    sessions = w.sessions(seed=0)
    assert all(s.carry_history and len(s.steps) == 6 for s in sessions)
    prefixes = {s.steps[0].messages[0]["content"] for s in sessions}
    assert len(prefixes) == 4
    assert all(st.messages[0]["role"] == "user" for s in sessions for st in s.steps[1:])


def test_w4_splits_json_and_tools():
    w = workloads.load(ROOT / "benchmarks/workloads/w4-structured-tools.toml")
    bodies = [s.steps[0].body for s in w.sessions(seed=0)]
    json_n = sum("response_format" in b for b in bodies)
    tools_n = sum("tools" in b for b in bodies)
    assert json_n + tools_n == len(bodies)
    assert 150 < json_n < 350
    assert bodies[0].get("response_format", {}).get("type", "json_schema") == "json_schema"


def test_w5_questions_are_parallel_single_token_requests():
    w = workloads.load(ROOT / "benchmarks/workloads/w5-kev-decisions.toml")
    for s in w.sessions(seed=0):
        assert s.parallel and 3 <= len(s.steps) <= 16
        contexts = {st.messages[0]["content"] for st in s.steps}
        assert len(contexts) == 1
        assert all(st.body["max_tokens"] == 1 and st.body["logprobs"] for st in s.steps)


def test_w6_mixes_embeddings_and_chat():
    w = workloads.load(ROOT / "benchmarks/workloads/w6-mixed-embed-generate.toml")
    apis = [s.steps[0].api for s in w.sessions(seed=0)]
    assert 150 < apis.count("embeddings") < 350
    assert apis.count("embeddings") + apis.count("chat") == len(apis)


def test_sharegpt_source_needs_a_dataset(tmp_path: Path):
    raw = tomllib.loads((ROOT / "benchmarks/workloads/w1-chat.toml").read_text())
    raw["prompt"]["source"] = "sharegpt"
    w = workloads.parse(raw)
    with pytest.raises(WorkloadError, match="--dataset"):
        w.sessions()
    data = tmp_path / "sg.json"
    data.write_text('[{"conversations": [{"from": "human", "value": "hello there"}, {"from": "gpt", "value": "hi"}]}]')
    prompts = workloads.load_sharegpt(data)
    assert prompts == ["hello there"]
    assert w.sessions(dataset=prompts)[0].steps[0].messages[-1]["content"] == "hello there"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(schema="other"), "schema"),
        (lambda r: r.update(status="supported-ish"), "status"),
        (lambda r: r.update(api="grpc"), "api"),
        (lambda r: r["load"].update(mode="burst"), "load.mode"),
        (lambda r: r["load"].update(levels=[]), "levels"),
        (lambda r: r["prompt"].update(tokens={"dist": "zipf"}), "unknown dist"),
        (lambda r: r["prompt"].update(tokens={"dist": "uniform", "min": 5, "max": 1}), "min 5 > max 1"),
        (lambda r: r.update(structured={"json_fraction": 0.5}), "response_format"),
        (lambda r: r.pop("prompt"), "missing 'prompt'"),
    ],
)
def test_invalid_workloads_are_rejected(mutate, message):
    raw = tomllib.loads((ROOT / "benchmarks/workloads/w1-chat.toml").read_text())
    mutate(raw)
    with pytest.raises(WorkloadError, match=message):
        workloads.parse(raw)


def test_dist_sampling_stays_in_bounds():
    import random

    rng = random.Random(0)
    d = Dist.parse({"dist": "lognormal", "median": 100, "sigma": 2.0, "min": 10, "max": 500}, "t")
    xs = [d.sample(rng) for _ in range(2000)]
    assert min(xs) == 10 and max(xs) == 500
    assert Dist.parse({"dist": "fixed", "value": 7}, "t").sample(rng) == 7
