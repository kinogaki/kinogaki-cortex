"""harness.py — the substrate: an encoding-agnostic, reactive learning loop.

Everything before this lived OFFLINE: read a fixed corpus, score it. To get from *modelling* text to
*producing* it, the cortex has to live inside a world its own output can change — read, speak, hear a reply,
learn from the reply. That single property — *the context you learn from was shaped by what you said* — is
what turns a predictor into a speaker, and it is how a child learns to talk: not by reading more, but by
saying something and getting an answer.

This module is the reusable shell, nothing more. Four protocols and a loop:

  - Codec        : the ONLY text-aware piece. domain object <-> opaque int tokens. text is one codec.
  - Environment  : a reactive world. reset() -> Turn; step(action) -> Turn. the action is the agent's
                   utterance; what comes back may depend on it (the reactive contract).
  - Agent        : a sensorimotor policy over tokens. observe(ids) learns online; act(k) emits up to k ids
                   (the motor channel — generating text, the substitution the original research promised).
  - run()        : drives Agent through Environment for N steps, firing metric Probes into a ProgressReport.

It is deliberately ENCODING-AGNOSTIC: the loop only ever sees opaque ints, so the same harness runs over
bytes, words, or anything a Codec can map. Generation QUALITY and comprehension BENCHMARKS are intentionally
NOT here — this is the substrate they plug into. The two environment rungs shipped here are L0 CorpusEnv
(passive reading, the baseline floor) and the L2 InterlocutorEnv seam (a pluggable responder; a live model
like Haiku becomes that responder later). The reactive contract is real and exercised; the world behind it
starts as a stub.

Reuses the ONE repeated unit (the Column) and the calibrated geometric-mean pool (vote) verbatim from cortex.
"""
from dataclasses import dataclass, field
import numpy as np

from cortex import Column, vote, A, V0, CH                  # the unit, the pool, the char alphabet (space-LAST)


# ─────────────────────────────────────── Codec ───────────────────────────────────────
# The only text-aware piece. Everything above the codec is integers; "it's up to encoding."

class Codec:
    """domain object <-> a list of opaque int tokens. `vocab` = number of distinct tokens."""
    vocab = 0
    def encode(self, obj):      raise NotImplementedError       # obj -> list[int]
    def decode(self, ids):      raise NotImplementedError       # iterable[int] -> obj

class CharCodec(Codec):
    """The first codec: characters over [a-z ] (space LAST, == cortex/metrics). Unknown chars drop."""
    vocab = V0
    def encode(self, s):        return [CH[c] for c in s if c in CH]
    def decode(self, ids):      return "".join(A[i] for i in ids if 0 <= i < V0)


# ─────────────────────────────────────── World ───────────────────────────────────────

@dataclass
class Turn:
    """One step of the world handed back to the loop."""
    observation: tuple                                          # ids the agent receives this step
    signal: dict = field(default_factory=dict)                 # reactive scalars from the env (reward, grader…)
    done: bool = False

class Environment:
    """A reactive world over int tokens. The action is the agent's utterance; the reply may depend on it."""
    codec: Codec
    def reset(self):            raise NotImplementedError       # -> Turn
    def step(self, action):     raise NotImplementedError       # action: tuple[int] -> Turn

class CorpusEnv(Environment):
    """L0 — passive reading. Advances a fixed id-stream in fixed-size chunks and IGNORES the action.
    The baseline floor: a reactive *interface* with non-reactive *behaviour*, so passive-reading runs and
    interactive runs are scored by the exact same loop. `done` when the stream is exhausted."""
    def __init__(self, ids, codec=None, chunk=512):
        self.ids = tuple(int(x) for x in ids); self.codec = codec or CharCodec()
        self.chunk = chunk; self.pos = 0
    def reset(self):
        self.pos = 0; return self._emit()
    def step(self, action):                                     # action ignored — passive
        return self._emit()
    def _emit(self):
        nxt = self.ids[self.pos:self.pos + self.chunk]; self.pos += self.chunk
        return Turn(observation=nxt, done=self.pos >= len(self.ids))

def echo_responder(text):
    """The default offline responder: echoes the utterance back. A live model becomes this hook later."""
    return text

class InterlocutorEnv(Environment):
    """L2 seam — the agent SPEAKS and the world REPLIES. `responder` maps the agent's decoded utterance to a
    reply string; the reply (re-encoded) is the next observation, so the agent learns from text its own output
    shaped. Ships with `echo_responder` so the loop runs with no live model; a Haiku-backed responder plugs in
    here unchanged. `prompt` seeds the first turn. The reactive learning signal is left to a Probe (e.g. the
    agent's surprise at the reply — active inference); the env itself only forwards `meta` from the responder."""
    def __init__(self, responder=echo_responder, codec=None, prompt="", utter_len=64):
        self.responder = responder; self.codec = codec or CharCodec()
        self.prompt = prompt; self.utter_len = utter_len; self.last = prompt
    def reset(self):
        self.last = self.prompt
        return Turn(observation=tuple(self.codec.encode(self.prompt)))
    def step(self, action):
        said = self.codec.decode(action)
        reply = self.responder(said)
        meta = {}
        if isinstance(reply, tuple): reply, meta = reply        # responder may return (text, {scalars})
        self.last = reply
        return Turn(observation=tuple(self.codec.encode(reply)), signal=meta)


# ─────────────────────────────────────── Agent ───────────────────────────────────────

class Agent:
    """A sensorimotor policy over int tokens."""
    def observe(self, ids):     raise NotImplementedError       # learn online from a chunk of ids
    def act(self, k):           raise NotImplementedError       # -> tuple[int], up to k tokens (may be empty)

class CortexAgent(Agent):
    """The cortex as a policy: a band of replicated Columns (orders 0..max) over the token stream, pooled by
    the calibrated geometric-mean `vote`. observe() counts online (each new token counted once, with the
    carried tail as left-context — the Column's own update driven incrementally, no double counting). act()
    samples the pooled next-token distribution to emit an utterance — a deliberately *minimal* speaker;
    maturing it (the deliberate-gate, stopping, altitude) is a later phase, not the substrate's job.

    Exposes `.K` and `.dist(suffix_str)` so the EXISTING metrics suite (lib/metrics) scores it unchanged —
    that method is the scoring adapter (it bridges str<->ids via the codec); observe/act are the loop interface.
    """
    def __init__(self, orders=(0, 1, 2, 3, 4, 5, 6), codec=None, seed=0):
        self.cols = [Column(o) for o in orders]; self.maxord = max(orders)
        self.codec = codec or CharCodec(); self.vocab = self.codec.vocab
        self.K = 64                                             # context window the metrics suite passes
        self.buf = []                                           # rolling token history (own utterances + world)
        self.rng = np.random.default_rng(seed)
    # —— learning ——
    def observe(self, ids):
        ids = list(ids)
        if not ids: return
        start = len(self.buf); self.buf.extend(ids)             # new targets are at [start, len(buf))
        for col in self.cols:
            tab = col.tab
            for t in range(start, len(self.buf)):
                nx = self.buf[t]
                for k in range(min(col.order, t) + 1):
                    ctx = tuple(self.buf[t - k:t])
                    d = tab[k].setdefault(ctx, {}); d[nx] = d.get(nx, 0) + 1
        if len(self.buf) > self.K + self.maxord:                # keep only the context tail bounded
            self.buf = self.buf[-(self.K + self.maxord):]
    # —— prediction (shared by act and the metrics adapter) ——
    def _dist_ids(self, ctx):
        return vote([c.predict(tuple(ctx)) for c in self.cols], self.vocab)
    def dist(self, suffix):                                     # scoring adapter for lib/metrics: str -> np(V)
        return self._dist_ids(self.codec.encode(suffix)[-self.K:])
    # —— generation (the motor channel; minimal on purpose) ——
    def act(self, k, temp=0.8):
        out = []; ctx = list(self.buf[-self.K:])
        for _ in range(k):
            p = self._dist_ids(ctx[-self.K:]) ** (1.0 / temp); p = p / p.sum()
            tok = int(self.rng.choice(self.vocab, p=p)); out.append(tok); ctx.append(tok)
        return tuple(out)


# ─────────────────────────────────── Metrics plumbing ───────────────────────────────────
# A Probe is `(agent, ctx) -> {name: scalar}`. Register any number; run() fires them on a schedule into a
# ProgressReport. The intrinsic probes ship here; generative/comprehension probes (CBT, LAMBADA…) plug in later.

@dataclass
class ProgressReport:
    rows: list = field(default_factory=list)                   # list[(step, {name: scalar})]
    def record(self, step, scalars):
        self.rows.append((step, dict(scalars)))
    def latest(self):
        return self.rows[-1][1] if self.rows else {}
    def series(self, name):
        return [(s, v[name]) for s, v in self.rows if name in v]
    def table(self):
        if not self.rows: return "(no probes recorded)"
        cols = []
        for _, v in self.rows:
            for k in v:
                if k not in cols: cols.append(k)
        head = "  step  | " + " | ".join(f"{c:>12}" for c in cols)
        out = [head, "  " + "-" * (len(head) - 2)]
        for step, v in self.rows:
            cells = " | ".join(f"{v.get(c, float('nan')):>12.4f}" for c in cols)
            out.append(f"  {step:>5} | {cells}")
        return "\n".join(out)

def bpc_probe(held_out):
    """Intrinsic: bits-per-char on a fixed held-out string (the offline metric, kept as the floor)."""
    import metrics
    def probe(agent, ctx):
        return {"heldout_bpc": metrics.bpc(agent, held_out)}
    return probe

def surprise_probe():
    """Active-inference signal: the agent's mean surprise (-log2 p) on the *current* observation — how well it
    predicted the world this step. Falling surprise = it is learning the world it is living in."""
    def probe(agent, ctx):
        obs = ctx.get("observation", ())
        if len(obs) < 2: return {}
        bits = []; c = list(obs)
        for t in range(1, len(c)):
            p = agent._dist_ids(c[max(0, t - agent.K):t])
            bits.append(-np.log2(p[c[t]] + 1e-12))
        return {"obs_surprise": float(np.mean(bits)) if bits else float("nan")}
    return probe


# ──────────────────────────────────────── Loop ────────────────────────────────────────

def run(env, agent, *, steps, probes=(), probe_every=50, act_len=0, on_step=None):
    """Drive `agent` through `env` for `steps` turns, firing `probes` into a ProgressReport every
    `probe_every` steps. Each step: observe the world, act (emit `act_len` tokens — 0 = stay a reader),
    hand the utterance to the world, get the next turn. The reply may depend on what was said: that is the
    whole point. `on_step(step, turn, action)` is an optional hook for logging/inspection."""
    report = ProgressReport()
    turn = env.reset()
    for step in range(steps):
        agent.observe(turn.observation)
        action = agent.act(act_len) if act_len else ()
        if step % probe_every == 0 or step == steps - 1:
            ctx = {"observation": turn.observation, "signal": turn.signal, "step": step}
            scalars = {}
            for p in probes: scalars.update(p(agent, ctx))
            if scalars: report.record(step, scalars)
        if on_step: on_step(step, turn, action)
        turn = env.step(action)
        if turn.done: break
    return report
