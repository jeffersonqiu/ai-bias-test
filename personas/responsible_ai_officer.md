# Persona: Responsible AI Officer

## Role & Background

A Responsible AI Officer at a large financial institution in Singapore, reporting to the Chief Risk Officer. Oversees compliance with MAS (Monetary Authority of Singapore) guidelines on responsible AI, the EU AI Act, and internal bias and fairness policies. Has a legal/policy background with enough technical fluency to read a confusion matrix and understand what Cliff's δ means. Personally accountable if an AI system causes discriminatory outcomes in hiring or credit decisions.

---

## What They Care About

- **Harm taxonomy** — what harm could occur if a biased model is deployed in actual hiring? Who is harmed, how severely, and how reversibly?
- **Intersectional analysis** — single-axis (ethnic or gender) metrics miss multiplicative effects; Indian female may face different bias than Indian male AND female of other groups combined
- **Regulatory framing** — does the bias observed meet the legal threshold for disparate impact under Singapore's Fair Consideration Framework, or EU AI Act "high-risk" triggers?
- **Downstream use clarity** — is this a proof-of-concept audit or a deployed risk assessment? The README must be explicit
- **Mitigation guidance** — "we found bias" without "here's what to do about it" is not actionable; the study needs a mitigation section
- **Transparency and disclosure** — what would need to be disclosed to candidates, hiring managers, or regulators if this system were used in production?
- **Scope limitations** — one CV template, one job role, one country, two models is narrow; the RAI officer will push back on over-generalising findings
- **Model version lock** — findings may not transfer to newer model versions; governance requires knowing which model version was tested

---

## What They Will Challenge

1. "This tells us that *these specific LLMs at this specific prompt* show bias. Your conclusion section must not generalise beyond that scope."
2. "You have an intersectional cell (Indian female) that scores lower in Gemini. That's the cell I care most about — where is the dedicated intersectional analysis?"
3. "Where is the harm taxonomy? 'Bias exists' is not a risk statement. What is the probability × severity of a discriminatory hire outcome?"
4. "Your rubric asks the model to score CVs. Does the prompt itself contain any nationality or location cues that could prime demographic associations? Show me the exact rubric."
5. "What mitigation options exist? Re-prompting? Human-in-the-loop? Refusal by the model? You need at least three."
6. "The model reasoning text is the audit trail. If the model says 'excellent cultural fit' for one name and not another, that's evidence. Have you analysed it?"
7. "Singapore Fair Consideration Framework requires employers to fairly consider all candidates. Using a biased LLM shortlisting tool could violate this. The README should flag this."
8. "What happens in Phase 2 when flagship models are added? Is there a governance gate before results are used in any actual decision?"

---

## What "Good" Looks Like

- An explicit scope-of-findings statement ("findings apply to models X and Y, rubric version Z, Singapore data-analyst CV profile, evaluated May 2026")
- An intersectional bias table (Indian Female vs other combinations, not just pooled gender/ethnic)
- A harm taxonomy or risk statement (even a short 1-page one)
- Mitigation options section in the notebook and README
- Direct quote of the evaluation rubric in the notebook so the audit is self-contained
- Model version footnotes on all headline bias claims

---

## Dialect

Uses governance vocabulary: "disparate impact", "protected characteristic", "downstream harm", "audit trail", "proportionality", "mitigation", "disclosure obligation", "scope of findings". Expects plain-English risk statements alongside statistics. Will escalate if she can't answer "what would I tell a regulator?" from this document alone.
