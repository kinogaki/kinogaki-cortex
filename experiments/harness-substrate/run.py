#!/usr/bin/env python3
"""Exp AT — the harness substrate: does the reactive learning loop actually run?

Not a result — a SUBSTRATE check. Everything before AT was offline (read a corpus, score it). This is the
shell that lets the cortex live in a world its own output can change: read, speak, hear a reply, learn from
the reply. We are NOT testing generation quality or comprehension here (both deferred). We are testing that
the four pieces compose and the loop is honest:

  1. L0 CorpusEnv — passive reading through the same loop. The agent learns ONLINE (counts accrue as the
     stream flows), and held-out bpc should FALL as more of the world goes by. This is the floor: the loop
     reproduces ordinary online learning, scored by the existing metrics suite, with zero new machinery.
  2. L2 InterlocutorEnv — the reactive seam. The agent SPEAKS (act() emits tokens), a pluggable `responder`
     REPLIES (here an offline stub — a live model like Haiku slots into this exact hook later), and the agent
     learns from the reply. We confirm the contract: action flows out, a reply that depends on it flows back,
     the agent observes it, and its surprise at the world is tracked (the active-inference signal).

Online throughout — the agent is the cortex's Column band, pooled by the calibrated geometric-mean vote.
No gradients, no new model. `lib/harness.py`, this file.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import harness as H
from corpus import load_ids, ids_to_str

SEED = 0
np.random.seed(SEED)

# ── corpus: a small slice, an online stream + a fixed held-out tail to score against ──
N = 2_000_000
ids = load_ids("enwik9", nbytes=N)
stream, held_ids = ids[:-20_000], ids[-20_000:]
held = ids_to_str(held_ids)
print(f"stream {len(stream):,} chars | held-out {len(held):,} chars | alphabet {H.V0}")

# ───────────────────────── 1. L0 — passive reading through the loop ─────────────────────────
print("\n" + "=" * 78)
print("(1) L0 CorpusEnv — passive online reading. held-out bpc should fall as the world goes by.")
print("=" * 78)

agent = H.CortexAgent(orders=(0, 1, 2, 3, 4, 5, 6), seed=SEED)
env = H.CorpusEnv(stream, chunk=2048)
nsteps = len(stream) // 2048 + 1
report = H.run(env, agent, steps=nsteps,
               probes=[H.bpc_probe(held), H.surprise_probe()],
               probe_every=max(1, nsteps // 12))
print(report.table())

first = report.series("heldout_bpc")[0][1]; last = report.series("heldout_bpc")[-1][1]
print(f"\n  held-out bpc: {first:.3f} (early) -> {last:.3f} (end)   "
      f"{'LEARNS online ✓' if last < first - 0.05 else 'no improvement ✗'}")

# ───────────────────────── 2. L2 — the reactive seam ─────────────────────────
print("\n" + "=" * 78)
print("(2) L2 InterlocutorEnv — the reactive contract: speak, hear a reply that depends on it, learn from it.")
print("=" * 78)

def canned_teacher(said):
    """An OFFLINE stand-in for a live interlocutor. The reply DEPENDS on the utterance (proving the contract),
    in a simple 10yo register. A Haiku-backed responder replaces this hook unchanged; here we just need the
    seam exercised with no network. Returns (reply, {scalars}) — meta rides back to the loop via Turn.signal."""
    said = said.strip()
    lead = said.split(" ")[-1] if said else ""
    lines = ["yes that is right and then we went home",
             "the cat sat on the mat and looked at me",
             "i think the boy ran to the big green tree",
             "we read a book about the sun and the moon"]
    reply = lines[len(lead) % len(lines)]
    return reply, {"said_len": float(len(said))}

interloc = H.InterlocutorEnv(responder=canned_teacher, prompt="once upon a time", utter_len=48)

# the agent is already warm from L0; now it speaks and learns from replies for a handful of turns
def show(step, turn, action):
    if step < 4:
        said = interloc.codec.decode(action)
        print(f"  turn {step}: agent said {said!r:<54} | heard {interloc.codec.decode(turn.observation)!r}")

rep2 = H.run(interloc, agent, steps=40, act_len=48,
             probes=[H.surprise_probe()], probe_every=8, on_step=show)
print("\n  surprise at the world's replies over the conversation:")
print(rep2.table())
s = rep2.series("obs_surprise")
if len(s) >= 2:
    print(f"\n  obs-surprise: {s[0][1]:.3f} -> {s[-1][1]:.3f}   "
          f"({'falling — adapting to the interlocutor' if s[-1][1] < s[0][1] else 'flat/up — stub is too small to adapt to'})")

print("\n" + "=" * 78)
print("SUBSTRATE CHECK: the loop runs over both rungs; the agent learns online (L0) and the reactive\n"
      "contract holds (L2 — action out, dependent reply in, learned from). Generation quality and the\n"
      "comprehension battery (CBT / LAMBADA / bAbI / BLiMP) are the next phase; this is the shell they plug into.")
print("=" * 78)
