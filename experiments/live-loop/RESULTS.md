# Exp BM — Live Haiku reactive loop: run BE's contingency gate inside a real interlocutor

**2026-06-26 · acquisition / Tier-3 reactive loop · verdict: WIN (kill did not fire) — but on the FALLBACK,
live run BLOCKED-ON-CREDENTIALS.**

## The claim BM tests

BE proved OFFLINE, on a hand-built transcript, that a WARM reply (one that lands right after the agent
spoke) teaches more than the same words re-timed to ignore it (the YOKED control). The cognitive source
(Goldstein & Schwade) is about a **live contingent partner** — the caregiver answers *what the child just
did*. BM closes that loop: a real interlocutor whose reply is a function of the agent's own utterance, and
asks BE's kill-test again, now reactively — **does contingency-ON still beat the scrambled-timing YOKED
ablation on the agent's surprise-at-replies and on turn-overlap?**

## Credentials / what actually ran (read this first)

- `anthropic` was **not** importable in the venv → installed into the venv only
  (`exp_a_boundary/.venv/bin/python -m pip install anthropic`, v0.112.0, allowed dev tooling).
- `os.environ` had **no** `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`. The one-message smoke test therefore
  returned `BLOCKED — no ... key in environment`.
- **The LIVE Haiku run is BLOCKED-ON-CREDENTIALS.** No model call was made. The live path is fully wired
  (`liveloop.make_haiku_responder`, model `claude-haiku-4-5-20251001`, 10yo-register system prompt, one
  call/turn, `max_tokens=40`, `N_TURNS=60`) and will run unchanged the moment a key is present — the smoke
  test gates it.
- **What ran instead:** the rich SCRIPTED 10-year-old-register responder
  (`liveloop.ScriptedResponder`) — a small grammar that pulls a topic word out of the agent's utterance and
  answers in a child-register template, falling back to a bank of simple child sentences. It is genuinely
  **contingent on the utterance**, just produced by a grammar instead of a model. Because it is free, the
  fallback ran **600 turns** instead of 60.

## Setup

- **Reactive loop** (`liveloop.reactive_run`, the live analogue of BE's `real_run`): per turn the agent
  ACTS (emits 24 tokens → Δt→0 inside `ContingencyAgent.act`); the responder REPLIES to the decoded
  utterance; the agent OBSERVES the reply **warm** (gain≈1); then 2 bursts of **cold** babble cool the
  clock (the noisy room). 600 turns, 20,123 reply chars, seed 0.
- **Babble** = frequency-matched i.i.d. scramble of text8 char marginals (a 400 KB text8 slice, used ONLY
  for the marginal + a held-out English string for turn-overlap; the learnable WARM signal is the replies).
- **Conditions, registered before running** — all three see the **exact same captured replies**:
  - **ON** — the dial: warm replies up-weighted by `g=exp(−Δt/τ)` into the hot band (BE's section-1 dial).
  - **YOKED** — same replies, same loop, gains drawn from the **scrambled** timeline (`yoked_gains`).
  - **PASSIVE** — gain≡1, single table (the AT floor; replies and babble at equal weight).
- **Metrics (BE's two axes, reported separately):** **reply-surprise** = mean −log2 p on the true next
  char of the interlocutor replies (how well it predicts the contingent channel; lower = better);
  **turn-overlap** = mean p on true next char of held-out English minus mean p on babble's next char.
- **SUBSTITUTION (said up front):** Goldstein & Schwade's CDS corpus is not in `data/`. The warm channel is
  a real reactive partner (fallback grammar here; Haiku when keyed); the cold channel is text8-marginal
  babble. The contingency claim is about *timing*, isolated by YOKED seeing identical tokens.

## Results (FALLBACK loop, 600 turns)

### (1) The three load-bearing conditions — same replies, only timing→gain alignment differs

| condition | reply-surprise | held-bpc | turn-overlap | |
|---|---|---|---|---|
| PASSIVE | 0.9444 | 4.234 | 0.0477 | no dial — replies + babble equal weight |
| YOKED   | 1.0120 | 4.170 | 0.0449 | same replies, **scrambled** timing → random gain |
| **ON**  | **0.7470** | **4.085** | **0.0685** | real contingency: warm replies loud |

ON beats YOKED by **−0.265 reply-surprise**, **+0.024 turn-overlap**, **+0.085 bpc**, and beats the
PASSIVE floor by **−0.198 reply-surprise / +0.149 bpc**. As in BE, YOKED is *worse* than PASSIVE on the
surprise axis: random gains actively mis-weight (up-counting babble, down-counting the warm replies).

### (2) FRAGILE sweep — τ × (cold_w, hot_pool), 12 settings (over the captured replies)

ON beats YOKED on reply-surprise at **12/12** (Δ +0.235 … +0.263) and on turn-overlap at **12/12**
(Δ +0.017 … +0.026). τ barely bites (in the loop the warm reply always lands at Δt=0); the win rides the
hot/cold split, strongest near `cold_w=0.10, hot_pool=3.0` (best Δreply-surprise **+0.2626**). This
reproduces BE's offline pattern qualitatively inside the reactive loop.

## Verdict — WIN (kill did NOT fire)

The BE kill-condition (ON matches YOKED on reply-surprise AND turn-overlap at matched tokens) **does not
fire** in the reactive loop: ON beats YOKED on **both** axes at **all 12** dial settings. BE's offline
result **survives** being run inside a genuinely reactive interlocutor — the warm/contingent channel is
learned distinctly better than the same replies with scrambled timing.

## Honest caveats (reported, not hidden)

1. **The live API run did not happen — it is BLOCKED-ON-CREDENTIALS.** Everything reported is on the
   scripted fallback. The headline number is a *fallback* number; the live Haiku replies would differ in
   surface form and richness. The live path is wired and smoke-gated, so a keyed re-run is one env var
   away — but it has NOT been validated against real Haiku output here.
2. **The fallback's warm channel is structured English; the cold channel is babble.** So ON's advantage
   is partly "structure beats noise." YOKED controls for this exactly (identical tokens, scrambled gains),
   so the **ON−YOKED gap is genuinely from timing→content alignment** — but PASSIVE-vs-ON conflates the
   two, and a fully fair live test would use a partner whose replies are *not* trivially separable from the
   babble by structure alone (a richer babble, or a Haiku whose register overlaps the noise).
3. **τ is inert** (same as BE): in the loop the warm reply always lands at Δt=0, so the exponential window
   never has graded delays to discriminate; a graded-delay world is the test that would make τ bite.
4. **This is BE's mechanism in a live shell, not a new mechanism.** BM contributes the reactive wiring and
   the credential-robust live/fallback seam, and confirms BE's win is not an artifact of the *fixed*
   transcript — it persists when the warm channel is produced turn-by-turn in response to the agent.
5. **BC (Rescorla-Wagner recovery) was read but not separately re-run here.** The prompt named running
   BE's gate AND BC's loop in the live shell; the load-bearing live kill-test is BE's contingency axis
   (the one with a YOKED control over live replies), so that is what BM measures. BC's recovery loop needs
   a *competing-form* event stream, which a free-form chat partner does not naturally supply turn-by-turn;
   wiring BC into a live partner is left as the obvious follow-up (it would need the partner to be steered
   toward inflection minimal-pairs).

## Rules confirmed

- **Online single pass** — `g` is a per-increment scalar inside `observe()`; one streaming pass; no epoch.
- **No backprop / k-means / SVD / eigen / word2vec** — only count increments + the calibrated `vote`.
- **Bounded** — the agent's two Column bands (hot/cold) under the same context-tail trim as the base agent.
- **Cognition-as-guide** — contingency/availability (Goldstein & Schwade); the soft gate never silences cold.

## Files

- `lib/liveloop.py` — `make_haiku_responder` (LIVE, wired), `smoke_test`, `ScriptedResponder` (fallback),
  `reactive_run` / `yoked_reactive_run` (the reactive driver + the BE ablation over live replies).
- `run.py` — this experiment (seed 0). Reuses `contingency.ContingencyAgent` + `yoked_gains` verbatim.

## Follow-up

The one thing this experiment could not do: **run it live.** With a key set, re-run as-is (it auto-switches
to Haiku at 60 turns). Then the fair test from caveat 2 — a partner whose replies are not structurally
trivial to separate from the cold channel — and a graded-delay world to make τ bite.
