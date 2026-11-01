"""Workload definitions (``benchmarks/workloads/*.toml``) and their deterministic expansion into request sessions.

A workload file describes the shape of the traffic, not a fixed request list: the same file, seed and dataset give the
same sessions for every backend, so Roost, vLLM, SGLang and llama.cpp receive identical requests. Synthetic prompts
are made of common English words; one word is about one token for the BPE tokenizers of the catalogue models, so
token counts are approximate. The server's ``usage`` counts are what the metrics use.
"""

from __future__ import annotations

import copy
import json
import math
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "roost-bench-workload/0"
STATUSES = {"supported", "unsupported", "experimental", "untested"}
APIS = {"chat", "completions", "embeddings", "mixed"}
LOAD_MODES = {"rate", "concurrency"}

# Common short English words; deterministic filler for synthetic prompts.
_WORDS = (  # noqa: SIM905 - one string reads better than a 260-line list
    "the of and to in is you that it he was for on are as with his they at be this have from or one had by word but "
    "not what all were we when your can said there use an each which she do how their if will up other about out many "
    "then them these so some her would make like him into time has look two more write go see number no way could "
    "people my than first water been call who oil its now find long down day did get come made may part over new "
    "sound take only little work know place year live me back give most very after thing our just name good sentence "
    "man think say great where help through much before line right too mean old any same tell boy follow came want "
    "show also around form three small set put end does another well large must big even such because turn here why "
    "ask went men read need land different home us move try kind hand picture again change off play spell air away "
    "animal house point page letter mother answer found study still learn should world high every near add food "
    "between own below country plant last school father keep tree never start city earth eye light thought head under "
    "story saw left few while along might close something seem next hard open example begin life always those both "
    "paper together got group often run important until children side feet car mile night walk white sea began grow"
).split()


class WorkloadError(ValueError):
    """A workload file that does not follow ``roost-bench-workload/0``."""


@dataclass(frozen=True)
class Dist:
    """An integer distribution: ``fixed``, ``uniform`` (inclusive) or ``lognormal`` (median, sigma), clamped."""

    kind: str
    value: int = 0
    lo: int = 1
    hi: int = 1 << 30
    median: float = 0.0
    sigma: float = 0.0

    @staticmethod
    def parse(spec: dict, where: str) -> Dist:
        kind = spec.get("dist")
        try:
            if kind == "fixed":
                return Dist("fixed", value=int(spec["value"]))
            if kind == "uniform":
                lo, hi = int(spec["min"]), int(spec["max"])
                if lo > hi:
                    raise WorkloadError(f"{where}: min {lo} > max {hi}")
                return Dist("uniform", lo=lo, hi=hi)
            if kind == "lognormal":
                return Dist(
                    "lognormal",
                    median=float(spec["median"]),
                    sigma=float(spec["sigma"]),
                    lo=int(spec.get("min", 1)),
                    hi=int(spec.get("max", 1 << 30)),
                )
        except KeyError as e:
            raise WorkloadError(f"{where}: missing {e.args[0]!r}") from None
        raise WorkloadError(f"{where}: unknown dist {kind!r} (fixed, uniform, lognormal)")

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return self.value
        if self.kind == "uniform":
            return rng.randint(self.lo, self.hi)
        x = round(math.exp(rng.gauss(math.log(self.median), self.sigma)))
        return min(max(x, self.lo), self.hi)


@dataclass
class Step:
    """One HTTP request of a session. The runner sets ``model``, ``messages`` and ``stream`` in ``body``."""

    api: str  # chat | completions | embeddings
    body: dict
    messages: list[dict] = field(default_factory=list)  # chat: messages appended for this step
    text: str = ""  # completions prompt or embeddings input


@dataclass
class Session:
    """Requests that belong together. ``carry_history`` makes chat steps see earlier turns and replies (W3).
    ``parallel`` sends the steps at once (W5's one-request-per-question equivalent)."""

    session_id: str
    steps: list[Step]
    carry_history: bool = False
    parallel: bool = False


@dataclass
class Workload:
    id: str
    title: str
    status: str
    api: str
    requests: int
    load_mode: str
    levels: list[float]
    raw: dict
    path: Path | None = None

    def sessions(self, seed: int = 0, dataset: list[str] | None = None) -> list[Session]:
        """Expand into ``requests`` sessions. Deterministic for a given seed and dataset."""
        return _expand(self, random.Random(f"{self.id}:{seed}"), dataset)


def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(max(n, 1)))


def load(path: str | Path) -> Workload:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    w = parse(raw, str(path))
    w.path = path
    return w


def parse(raw: dict, where: str = "<workload>") -> Workload:
    if raw.get("schema") != SCHEMA:
        raise WorkloadError(f"{where}: schema must be {SCHEMA!r}")
    for key in ("id", "title", "status", "api", "requests", "load", "prompt"):
        if key not in raw:
            raise WorkloadError(f"{where}: missing {key!r}")
    if raw["status"] not in STATUSES:
        raise WorkloadError(f"{where}: status {raw['status']!r} not one of {sorted(STATUSES)}")
    if raw["api"] not in APIS:
        raise WorkloadError(f"{where}: api {raw['api']!r} not one of {sorted(APIS)}")
    load_ = raw["load"]
    if load_.get("mode") not in LOAD_MODES:
        raise WorkloadError(f"{where}: load.mode must be one of {sorted(LOAD_MODES)}")
    levels = load_.get("levels")
    if not levels or any(not isinstance(v, int | float) or v <= 0 for v in levels):
        raise WorkloadError(f"{where}: load.levels must be a non-empty list of positive numbers")
    if load_["mode"] == "concurrency" and any(int(v) != v for v in levels):
        raise WorkloadError(f"{where}: concurrency levels must be integers")
    Dist.parse(raw["prompt"].get("tokens", {}), f"{where}: prompt.tokens")
    if raw["api"] != "embeddings":
        Dist.parse(raw.get("output", {}).get("tokens", {}), f"{where}: output.tokens")
    if "decision" in raw:
        Dist.parse(raw["decision"].get("questions", {}), f"{where}: decision.questions")
    structured = raw.get("structured", {})
    if float(structured.get("json_fraction", 0)) > 0 and "response_format" not in structured:
        raise WorkloadError(f"{where}: structured.json_fraction needs structured.response_format")
    if float(structured.get("tools_fraction", 0)) > 0 and "tools" not in structured:
        raise WorkloadError(f"{where}: structured.tools_fraction needs structured.tools")
    if float(structured.get("json_fraction", 0)) + float(structured.get("tools_fraction", 0)) > 1:
        raise WorkloadError(f"{where}: structured fractions add up to more than 1")
    if raw["prompt"].get("source", "synthetic") not in {"synthetic", "sharegpt"}:
        raise WorkloadError(f"{where}: prompt.source must be synthetic or sharegpt")
    return Workload(
        id=raw["id"],
        title=raw["title"],
        status=raw["status"],
        api=raw["api"],
        requests=int(raw["requests"]),
        load_mode=load_["mode"],
        levels=[float(v) for v in levels],
        raw=raw,
    )


def load_sharegpt(path: str | Path) -> list[str]:
    """First human turn of each conversation in a ShareGPT-format JSON file (not committed; given at run time)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human" and turn.get("value", "").strip():
                prompts.append(turn["value"])
                break
    if not prompts:
        raise WorkloadError(f"{path}: no human turns found")
    return prompts


def _expand(w: Workload, rng: random.Random, dataset: list[str] | None) -> list[Session]:
    raw = w.raw
    prompt_dist = Dist.parse(raw["prompt"]["tokens"], "prompt.tokens")
    out_dist = Dist.parse(raw["output"]["tokens"], "output.tokens") if "output" in raw else None
    source = raw["prompt"].get("source", "synthetic")
    if source == "sharegpt" and not dataset:
        raise WorkloadError(f"{w.id}: prompt.source is sharegpt; pass the dataset file with --dataset")
    extra = dict(raw.get("body", {}))
    prefix = raw.get("prefix", {})
    shared = int(prefix.get("shared_tokens", 0))
    groups = [_words(rng, shared) for _ in range(int(prefix.get("groups", 1)))] if shared else []
    turns = int(prefix.get("turns", 1))
    structured = raw.get("structured", {})
    decision = raw.get("decision")
    mix = raw.get("mix", {})

    def user_text() -> str:
        if source == "sharegpt":
            return rng.choice(dataset)
        return _words(rng, prompt_dist.sample(rng))

    def gen_body() -> dict:
        body = copy.deepcopy(extra)
        if out_dist is not None:
            body["max_tokens"] = out_dist.sample(rng)
        return body

    sessions: list[Session] = []
    for i in range(w.requests):
        sid = f"{w.id}-{i:05d}"
        api = w.api
        if api == "mixed":
            api = "embeddings" if rng.random() < float(mix.get("embed_fraction", 0.5)) else "chat"
        system = [{"role": "system", "content": rng.choice(groups)}] if groups else []
        if api == "embeddings":
            sessions.append(Session(sid, [Step("embeddings", {}, text=user_text())]))
        elif api == "completions":
            text = (system[0]["content"] + "\n\n" if system else "") + user_text()
            sessions.append(Session(sid, [Step("completions", gen_body(), text=text)]))
        elif decision is not None:
            q_dist = Dist.parse(decision["questions"], "decision.questions")
            context = system or [{"role": "system", "content": _words(rng, prompt_dist.sample(rng))}]
            body = copy.deepcopy(decision.get("body", {}))
            body.setdefault("max_tokens", 1)
            question = Dist.parse(decision.get("question_tokens", {"dist": "fixed", "value": 32}), "question_tokens")
            steps = [
                Step(
                    "chat",
                    copy.deepcopy(body),
                    messages=context + [{"role": "user", "content": _words(rng, question.sample(rng))}],
                )
                for _ in range(q_dist.sample(rng))
            ]
            sessions.append(Session(sid, steps, parallel=bool(decision.get("parallel", True))))
        else:
            steps = []
            for t in range(turns):
                body = gen_body()
                roll = rng.random()
                if roll < float(structured.get("json_fraction", 0)):
                    body["response_format"] = copy.deepcopy(structured["response_format"])
                elif roll < float(structured.get("json_fraction", 0)) + float(structured.get("tools_fraction", 0)):
                    body["tools"] = copy.deepcopy(structured["tools"])
                    body["tool_choice"] = structured.get("tool_choice", "auto")
                messages = (system if t == 0 else []) + [{"role": "user", "content": user_text()}]
                steps.append(Step("chat", body, messages=messages))
            sessions.append(Session(sid, steps, carry_history=turns > 1))
    return sessions
