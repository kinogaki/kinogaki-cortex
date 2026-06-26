# Experiment BJ — structure-graded recursion exposure (self-gated embedding depth)

The one curriculum AK did not test. AK staged the *memory budget* (leak-horizon) and found growing it ties
full-from-start — a count learner has no gradient to lock, so "starting small" rescues nothing. BJ stages
**structure** instead: order the stream by embedding **depth**, self-gated on the agent's own branching
entropy (admit depth d+1 only once depth-d transition entropy stabilizes — teacher-free, reads its own
counts). The test is center-embedded subject–verb agreement (`S1 S2 … run run`, the outer verb agrees with
the outer subject across the nest); the recursion-only axis is perplexity on the OUTER (deep) agreement char
at depth 2/3. Verdict: an **honest negative** — graded ordering ties full-from-start (mean −0.02%, inside the
noise band) on the deep token, and the reversed (deep-first) order ties too. Depth-1 is learned by every
regime (acc 0.78); depth-2/3 stay at chance (acc ~0.51) for every regime, because a windowed count learner
has no stack to carry the shallow skill across the embedding. AK extends to structural ordering — a clean,
expected, publishable negative. `lib/recursion.py` holds the reusable corpus + self-gated curriculum.
