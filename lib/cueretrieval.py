"""Cue-based retrieval with similarity-based (fan) interference — content-addressable memory.

Offset-attention (Exp S) keys memory by relative POSITION: "what followed the word d steps back?".
That key is a fixed integer, so it can only reach a fixed window and its informativeness DECAYS with d.
A dependent and its antecedent can sit arbitrarily far apart (a subject and its verb across an embedded
clause), and no position-key reaches that far reliably. Human sentence processing solves this with
CONTENT-ADDRESSABLE RETRIEVAL (Lewis & Vasishth 2005; Jaeger/Engelmann/Vasishth 2017): a dependent fires
a set of retrieval CUES (a feature bundle — word class, number/agreement, recency), and every item in
memory whose features match the cues is reactivated. The item that wins is the one with the highest
ACTIVATION, and the model's signature prediction is:

    activation(item) = leaked_base(item)  /  FAN(cue)

where leaked_base is a recency-leaked count of how strongly the item is currently in memory, and FAN is
the number of stored items that ALSO match the cue. Sharing a cue with distractors DIVIDES activation —
that is similarity-based interference (the fan effect, Anderson 1974): retrieval degrades and mis-targets
when a distractor shares the dependent's cues. This is the count-based, no-backprop form of "reaching
back by the right cue" — generalize offset-attention's position key to a {feature} key, and weight by
fan instead of by a fixed offset's information gain.

  RetrievalStore(half_life, match)        a leaky content-addressable memory of feature-keyed items.
    .observe(features, item, t)           store/refresh an item with its feature bundle at time t.
    .retrieve(cues, t)                     return ranked [(item, activation), ...] matching the cues.
    .fan(cue, t)                           number of currently-live items sharing a single cue.

The store is ONLINE single-pass, bounded (a leaky decay + a cap evicts the stalest items), and keeps
only counts and timestamps — no gradients, no batch optimisation. The cue->items index is the count
table; the leak is recency; the fan division is the interference law. Everything is read off counts.
"""
import math
from collections import defaultdict, OrderedDict


# ---- feature derivation: count-based / morphological, no linguistic resources --------------------
# We need a feature bundle per word. Keep it transparent and corpus-derived. Number (sing/plur) is the
# agreement-bearing feature we test; word-class is a coarse function/content split. Both are heuristics
# read off the surface form + a frequency table — no POS tagger, in keeping with the substrate.

# Agreement-bearing copulas/auxiliaries, split by number. These are the words whose form REVEALS the
# number of their subject, so they are the cleanest agreement probe in raw text (Lewis & Vasishth use
# exactly such forms). Strings here are matched against the decoded word.
SING_VERBS = {"is", "was", "has", "does"}
PLUR_VERBS = {"are", "were", "have", "do"}

# Determiners that mark the number of the noun they introduce — used to label a noun's number reliably
# without a tagger ("a/an/this/that" -> singular head, "these/those/many/several" -> plural head).
SING_DET = {"a", "an", "this", "that", "one", "each", "every"}
PLUR_DET = {"these", "those", "many", "several", "few", "both", "two", "three"}

# Closed-class function words: if a token is here it's FUNCTION, else CONTENT. A coarse word-class cue.
FUNCTION = {
    "the", "a", "an", "of", "to", "in", "and", "or", "but", "for", "with", "on", "at", "by", "from",
    "as", "that", "this", "these", "those", "it", "its", "he", "she", "they", "we", "you", "i", "his",
    "her", "their", "our", "your", "my", "is", "was", "are", "were", "has", "have", "had", "be", "been",
    "do", "does", "did", "not", "no", "so", "if", "than", "then", "into", "over", "under", "which",
    "who", "what", "when", "where", "while", "about", "after", "before", "between", "through",
}


def number_of(word):
    """singular | plural | None  for a NOUN, from its surface form (the -s plural heuristic).
    Conservative: only commits when the morphology is clear, else None (so we never force a label)."""
    if len(word) < 3:
        return None
    if word in FUNCTION:
        return None
    if word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return "singular"          # -ss/-us/-is are singular endings (class, virus, basis), not plurals
    if word.endswith("s"):
        return "plural"
    return "singular"


def verb_number(word):
    """singular | plural | None — the number a copula/aux AGREES with (the subject's number)."""
    if word in SING_VERBS:
        return "singular"
    if word in PLUR_VERBS:
        return "plural"
    return None


def word_class(word):
    return "function" if word in FUNCTION else "content"


# ---- the leaky content-addressable retrieval store -----------------------------------------------

class RetrievalStore:
    """A content-addressable memory of feature-keyed items with recency-leaked activation and a fan law.

    half_life:   in 'time' units (we tick once per word); base activation halves every half_life words.
    recency_cap: bounded memory — each cue keeps at most this many recent items; the stalest is dropped.

    Items are stored under EACH of their features (a posting list per feature value, e.g. ('number',
    'singular') -> [item, item, ...]). A retrieval cues a SET of (feature,value) pairs; an item matches
    if it carries all the cued features (AND) — or, in soft mode, scores by fraction of cues matched.

    The store is RECENCY-BOUNDED: each posting list keeps only the most-recent `recency_cap` items for a
    cue. This makes it both bounded in memory AND cheap (fan/retrieve scan a small recent set), and it is
    the cognitively-right bound — interference in Lewis & Vasishth comes from RECENTLY-active competitors,
    not from every item ever seen. An item whose leaked activation has fallen below `floor` is functionally
    dead and is pruned lazily when its posting list is scanned."""

    def __init__(self, half_life=25.0, recency_cap=256, floor=0.02):
        self.lam = math.log(2) / half_life       # leak rate: exp(-lam * age) halves every half_life
        self.cap = recency_cap                    # max live items kept per cue (the bounded memory knob)
        self.floor = floor                        # activation below this = dead (prune / ignore)
        self.index = defaultdict(OrderedDict)     # (feat,val) -> OrderedDict{item_id: rec} recency-ordered
        self.t = 0

    def observe(self, features, item, t=None):
        """Store/refresh `item` (any hashable) with its feature bundle (iterable of (feat,val) pairs).
        Re-observing refreshes recency and bumps the hit count (a repeated mention is a stronger memory).
        Each cue's posting list is move-to-end on refresh and capped to the most-recent `cap` items."""
        if t is None:
            t = self.t
        rec = {"t": t, "n": 1, "item": item}
        for fv in features:
            post = self.index[fv]
            old = post.get(item)
            if old is not None:
                old["t"] = t
                old["n"] += 1
                post.move_to_end(item)
            else:
                post[item] = rec
                if len(post) > self.cap:
                    post.popitem(last=False)      # drop the stalest item for this cue (O(1))

    def tick(self):
        self.t += 1

    def _act(self, rec, t):
        """Leaked base activation: hit-count * recency-leak (halves every half_life)."""
        return rec["n"] * math.exp(-self.lam * (t - rec["t"]))

    def _live(self, cue, t):
        """Yield (item, rec, activation) for the LIVE items under a single cue (activation >= floor),
        pruning dead ones lazily. Posting lists are recency-ordered, so the live items are the tail."""
        post = self.index.get(cue)
        if not post:
            return
        dead = []
        for item, rec in reversed(post.items()):
            a = self._act(rec, t)
            if a < self.floor:
                dead.append(item)
                continue
            yield item, rec, a
        for item in dead:
            post.pop(item, None)

    def fan(self, cue, t=None):
        """How many currently-LIVE items share this single cue? The interference count (Anderson's fan)."""
        if t is None:
            t = self.t
        return sum(1 for _ in self._live(cue, t))

    def retrieve(self, cues, t=None, soft=False, topn=8):
        """Cue a set of (feature,value) pairs; return ranked [(item, activation), ...].

        activation(item) = leaked_base(item) / FAN(cue), the Lewis & Vasishth law. The fan dividing an
        item is the MAX fan over the item's matched cues (the most-confusable cue dominates interference).
        Hard mode: an item must carry ALL cued features. Soft mode: partial matches allowed, scored by the
        matched fraction (graded content-addressability). Returns the top-n by activation."""
        if t is None:
            t = self.t
        cues = list(cues)
        if not cues:
            return []
        live = {cue: dict((it, (rec, a)) for it, rec, a in self._live(cue, t)) for cue in cues}
        fan_cache = {cue: max(1, len(live[cue])) for cue in cues}
        # candidate items = union of live items across cues
        cand = {}
        for cue in cues:
            cand.update(live[cue])
        out = []
        for it, (rec, _) in cand.items():
            matched = [cue for cue in cues if it in live[cue]]
            if not soft and len(matched) < len(cues):
                continue
            base = self._act(rec, t)
            fan = max(fan_cache[cue] for cue in matched)
            act = base / fan
            if soft:
                act *= len(matched) / len(cues)
            out.append((it, act))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out[:topn]
