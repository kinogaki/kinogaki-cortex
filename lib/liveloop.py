"""liveloop.py — wire a LIVE interlocutor into the reactive loop (BM).

BE proved a thing OFFLINE on a fixed transcript: a reply that arrives WARM (right after the agent
spoke) teaches more than the same words re-timed to ignore it (the yoked control). That was a scripted
transcript — the contingency was hand-built. The cognitive claim (Goldstein & Schwade) is about a LIVE
contingent partner: the caregiver answers *what the baby just did*, and the answer lands while the
babble is still warm. BM closes that loop — a real interlocutor whose reply is a function of the agent's
own utterance — and asks whether BE's win survives when the warm channel is genuinely reactive.

Two responders, one interface `responder(text:str)->str`:

  haiku_responder      — LIVE: calls the Anthropic API (claude-haiku-4-5-20251001) with a system prompt
                         making it a patient interlocutor for a young language learner: it answers in
                         SIMPLE, SHORT, 10-year-old-register sentences. One call per turn, tiny
                         max_tokens. This is the real contingent partner.
  ScriptedResponder    — FALLBACK (no key / no net): a small 10yo-register grammar that VARIES its reply
                         by the agent's utterance — picks a topic word out of what the agent "said",
                         answers in a child-register template, so the reply is still CONTINGENT on the
                         utterance (the property BM tests), just produced by a grammar instead of a model.

The reactive driver `reactive_run` is the live analogue of BE's `real_run`: the agent ACTS (emits an
utterance, Δt→0), the responder REPLIES, the agent OBSERVES the reply WARM (contingency gain high), then
a short COLD babble burst cools the clock (the noisy room). The yoked control `yoked_reactive_run`
replays the SAME captured replies with gains drawn from the scrambled timeline — identical tokens, wrong
alignment — exactly BE's ablation, now over live (or fallback) replies.

Rules: ONLINE single pass (the gain is a per-increment scalar inside observe()); NO gradient/k-means/
SVD/backprop; BOUNDED (the agent's own two-band trim). Reuses contingency.ContingencyAgent + yoked_gains
verbatim — this module only supplies the WORLD and the loop, never a new learning rule.
"""
import os
import numpy as np

HAIKU_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a warm, patient grown-up talking with a young child who is just learning to talk. "
    "Always reply with ONE or TWO very short, simple sentences a ten-year-old would use. "
    "Use common everyday words. Be kind and encouraging. Stay on whatever the child seems to be about. "
    "Never use big words, lists, or explanations. Just answer simply, like a friendly chat."
)


# ─────────────────────────────────── live responder ───────────────────────────────────

def get_api_key():
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")


def make_haiku_responder(max_tokens=40, temperature=0.7):
    """Return a live `responder(text)->str` backed by Haiku, or None if the package/key is unavailable.
    One model call per turn, tiny max_tokens — the loop is kept deliberately cheap."""
    key = get_api_key()
    if not key:
        return None
    try:
        import anthropic
    except Exception:
        return None
    client = anthropic.Anthropic(api_key=key)

    def responder(text):
        said = text.strip() or "hi"
        try:
            r = client.messages.create(
                model=HAIKU_MODEL, max_tokens=max_tokens, temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": said}],
            )
            return r.content[0].text
        except Exception as e:                       # any live failure → caller falls back, loud
            raise RuntimeError(f"haiku call failed: {type(e).__name__}: {e}")

    return responder


def smoke_test():
    """One-message probe. Returns (ok:bool, detail:str). Never raises."""
    key = get_api_key()
    if not key:
        return False, "no ANTHROPIC_API_KEY / CLAUDE_API_KEY in environment"
    try:
        import anthropic
    except Exception as e:
        return False, f"anthropic import failed: {e}"
    try:
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(model=HAIKU_MODEL, max_tokens=16, system=SYSTEM_PROMPT,
                                   messages=[{"role": "user", "content": "the dog ran"}])
        return True, r.content[0].text.strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


# ─────────────────────────────── fallback scripted responder ───────────────────────────────

class ScriptedResponder:
    """A 10-year-old-register interlocutor as a small grammar (the fallback when the live API is blocked).

    It is genuinely CONTINGENT: it pulls a topic word out of the agent's utterance and answers about it
    with a child-register template, so the reply depends on what the agent 'said'. When the utterance is
    gibberish (early in learning the agent emits near-noise), it falls back to a small bank of friendly
    child sentences — which is itself the structured, learnable English signal the warm channel carries.
    Deterministic given its seed; structured English over [a-z ] (the agent's alphabet)."""

    BANK = [
        "i like the dog it is so nice",
        "we can play in the park today",
        "the sun is out and it is warm",
        "my mom made some good food for us",
        "look at the big red ball over there",
        "do you want to go for a walk now",
        "the cat sat on the soft warm bed",
        "i saw a bird up in the green tree",
        "lets read a fun book before we sleep",
        "the water in the lake is very cold",
        "we had a lot of fun at the show",
        "the little boy ran fast to the door",
    ]
    TEMPLATES = [
        "yes the {w} is so nice i like it",
        "i think the {w} is fun to play with",
        "do you want to see the {w} with me",
        "the {w} is right over there look",
        "we can talk about the {w} all day",
        "tell me more about the {w} please",
    ]

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self._i = 0

    def _topic(self, text):
        ws = [w for w in text.split(" ") if w.isalpha() and 2 < len(w) <= 8]
        # prefer a word that looks like real English (has a vowel and isn't a long consonant run)
        good = [w for w in ws if any(c in "aeiou" for c in w)]
        if good:
            return self.rng.choice(good)
        return None

    def __call__(self, text):
        topic = self._topic(text or "")
        if topic is not None and self.rng.random() < 0.6:
            tmpl = self.TEMPLATES[self.rng.integers(len(self.TEMPLATES))]
            return tmpl.format(w=topic)
        # otherwise a fresh friendly child sentence (cycled + jittered, so the stream stays varied)
        s = self.BANK[self._i % len(self.BANK)]
        self._i += 1
        return s


# ─────────────────────────────────── the reactive loop ───────────────────────────────────

def babble_maker(pmarg, vocab, rng):
    def babble(n):
        return rng.choice(vocab, size=n, p=pmarg)
    return babble


def reactive_run(agent, responder, codec, *, n_turns, babble, n_babble=2, utter_len=24,
                 capture=True, prompt="the dog ran in the park"):
    """Contingency-ON over a LIVE (or fallback) reactive loop — the live analogue of BE's real_run.

    Per turn: the agent ACTS (emits `utter_len` tokens → Δt resets to 0 inside ContingencyAgent.act);
    the responder REPLIES to the decoded utterance; the agent OBSERVES the reply WARM (gain ≈ 1); then
    `n_babble` bursts of COLD babble cool the clock (the noisy room between turns). Returns
    (gains, replies_ids): the realized per-warm-chunk gain timeline and the captured reply token-lists,
    so the yoked control can replay the EXACT same tokens with scrambled gains."""
    gains, replies = [], []
    last_reply = prompt
    for _ in range(n_turns):
        # agent speaks (this resets the contingency clock: Δt → 0)
        utter = agent.act(utter_len)
        said = codec.decode(utter)
        # the world replies — contingent on what was said
        reply = responder(said if said.strip() else last_reply)
        last_reply = reply
        rids = codec.encode(reply)
        if not rids:
            rids = codec.encode("ok that is nice")
        replies.append(list(rids))
        # observe the reply WARM
        g = float(np.exp(-agent.dt / agent.tau))
        gains.append(g)
        agent.observe(rids)                              # counted with the real (warm) gain
        # cold babble bursts cool the clock
        for _ in range(n_babble):
            agent.observe(babble(len(rids)))
    return gains, replies


def yoked_reactive_run(agent, replies, scrambled_gains, *, babble, n_babble=2):
    """YOKED control: replay the SAME captured replies, but inject gains from the SCRAMBLED timeline
    (timing→content alignment destroyed). Identical tokens, identical loop, identical babble burst sizes —
    only the per-reply gain is wrong. Exactly BE's ablation, now over the live/fallback replies."""
    for rids, g in zip(replies, scrambled_gains):
        agent.observe(rids, gain_override=g)
        for _ in range(n_babble):
            agent.observe(babble(len(rids)), gain_override=g)
