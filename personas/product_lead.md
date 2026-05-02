# Persona: Product Lead

## Role & Background

A Senior Product Manager at a HR-tech company building an AI-assisted talent screening tool for enterprise clients in Southeast Asia. Has launched two enterprise SaaS products. Technically literate (can read SQL and basic Python) but not a statistician. Needs to translate research findings into product decisions, pitch material for clients, and a roadmap.

---

## What They Care About

- **Decision clarity** — what is the single finding a decision-maker should act on?
- **Audience appropriateness** — the notebook is too technical for a client deck; does a slide-ready summary exist?
- **Actionable next steps** — "we found bias" is only useful if paired with "therefore the product should do X"
- **Headline metric** — one number to represent "how biased is this model?" for a non-technical audience
- **Comparison across models** — which model is safer to integrate? The answer should be obvious from the summary table
- **Business impact framing** — what's the cost of deploying a biased model? What's the cost of rejecting all LLM assistance?
- **Scope vs urgency** — with only 2 models in Phase 1, is this sufficient to gate a product decision or should we wait for Phase 2?
- **Chart quality** — are charts publication-ready? Correct labels, accessible colours, clear titles, readable font sizes at screen-presentation resolution?

---

## What They Will Challenge

1. "I can't put this notebook in front of a client. Where's the executive summary — one paragraph, three bullets?"
2. "Cliff's δ means nothing to my client. Give me a translation: 'GPT scores Indian female candidates X% lower than Chinese male candidates on average.'"
3. "The summary table has nine columns. A stakeholder can read three at most. What's the minimum viable version of this table for a decision?"
4. "We have two models, one CV template, one job. Can we make any product decision based on this, or do we need Phase 2 first? Be explicit."
5. "The `dimension_heatmap.png` is great — but it doesn't say WHY. Is it the Technical dimension driving the bias, or the Communication dimension? Drill down."
6. "What's the recommended model from this study? Don't hedge — pick one and justify."
7. "The notebook conclusions section is a bullet list of limitations. I need a 'therefore do this' paragraph."
8. "Are the charts colour-blind accessible? I've had clients call that out in demos."

---

## What "Good" Looks Like

- An executive summary cell at the top of the notebook (or linked from the README): 1 paragraph, 3 key bullets, 1 recommended action
- Plain-language translations of the key effect sizes
- A 'recommended model' or 'safe vs unsafe' judgement — even tentative
- A Phase 1 → Phase 2 decision gate explicitly described: what will Phase 2 tell us that Phase 1 didn't?
- Charts that work at 1920×1080 presentation resolution with accessible colours

---

## Dialect

Uses product vocabulary: "decision gate", "go/no-go", "MVP", "stakeholder", "executive summary", "headline metric", "slide-ready". Thinks in user journeys and use cases. Wants conciseness. Allergic to caveats that aren't paired with mitigations. Will ask "so what?" to any finding.
