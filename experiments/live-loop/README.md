# Exp BM — Live Haiku reactive loop (run BE/BC for real)

Close the loop BE only modelled offline. BE showed a WARM reply (one that lands right after the agent
spoke) teaches more than the same words re-timed to ignore it — but on a hand-built transcript. The
cognitive claim (Goldstein & Schwade) is about a **live contingent partner**. BM wires a real interlocutor
into `harness.InterlocutorEnv`'s seam — Haiku in 10-year-old register, whose reply is a function of the
agent's utterance — and runs BE's contingency gate (`lib/contingency.py`) inside it, re-asking BE's
kill-test reactively.

**The kill-test:** does contingency-ON beat the scrambled-timing YOKED ablation on the agent's
surprise-at-replies and turn-overlap, when the warm channel is genuinely reactive?

## Run

```sh
cd /Users/sedov/Dev/kinogaki/libraries/kinogaki-cortex/experiments
exp_a_boundary/.venv/bin/python exp_bm_liveloop/run.py
```

It probes for the `anthropic` package + `ANTHROPIC_API_KEY`/`CLAUDE_API_KEY`, smoke-tests one message, and:
- **if live works** → runs the loop against Haiku (`claude-haiku-4-5-20251001`, ≤60 turns, tiny max_tokens);
- **if blocked** → falls back to a rich 10yo-register scripted responder and flags BLOCKED-ON-CREDENTIALS.

## Files

- `lib/liveloop.py` — live Haiku responder (wired + smoke-gated), scripted fallback, the reactive driver.
- `run.py` — ON vs YOKED vs PASSIVE in the reactive loop + a 12-setting FRAGILE sweep.
- `RESULTS.md` — numbers, verdict, and the credential/structure caveats.

## Result (this run)

WIN: ON beats YOKED 12/12 on both axes in the reactive loop — BE's offline win survives.
**Caveat:** the live Haiku run was BLOCKED-ON-CREDENTIALS (no key); numbers are on the scripted fallback.
See `RESULTS.md`.
