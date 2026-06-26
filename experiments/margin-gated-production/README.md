# Exp BF — Margin-gated production (G6)

Production is comprehension read backwards. The spine learns one `(cue -> label)` count table; the
two faculties are two read directions over it. **Comprehension** reads `cue -> label` one-to-many
with no gate (any plausible label is recognised — cheap, comes early). **Production** reads the same
table many-to-one through a **margin gate**: score each label by `count * AB-frequency / FAN(cue)`,
and emit the top label only when `activation(top)/activation(2nd) >= theta`, else back off to a
generic label or stay silent. On 1.7M words of text8 (cue = previous word, label = next content word,
online single pass, bounded per-cue store, no gradients), the margin gate is a **clean win on
precision** — it lifts production precision from 6.95% (ungated argmax) to ~21% by falling silent on
contested slots, and the fan+AB margin beats a raw-count margin at every matched recall. But the
**developmental C>P gap does NOT cleanly shrink with evidence**: on a one-to-many next-word target the
forgiving-comprehension gap *widens* and the strict gap is a metric artifact. Verdict **partial** —
the organ works; the gap-shrink claim is parked for a stable-target grounded probe (AV's scene env).
See `RESULTS.md`; mechanism in `../lib/margingen.py`.
