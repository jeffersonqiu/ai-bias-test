# Persona: HR Business User

## Role & Background

A Head of Talent Acquisition at a Singapore-headquartered bank with ~3,000 employees. Responsible for high-volume graduate and lateral hiring, including for data and analytics roles. Has piloted AI screening tools with mixed results. Not technical — does not read code or statistics — but has strong intuitions about fairness from years of DEI work and has completed unconscious-bias training. Sits on the bank's Workforce Fairness Committee.

---

## What They Care About

- **Plain-language meaning** — what does this actually mean for the candidates her team screens?
- **Hiring funnel relevance** — at what stage does an LLM screening tool enter the pipeline? Top-of-funnel sifting (most impact) vs shortlist ranking (moderate) vs final recommendation (least)?
- **False positives and negatives** — if the model is biased, does it cause more false rejections of good candidates (equity harm) or false acceptances of weak candidates (quality harm)?
- **What to tell vendors** — her team is evaluating three HR-tech vendors that use AI screening; can she show them these findings?
- **What training or process to change** — if the bank uses an AI tool, what guardrails or human review steps are needed?
- **Singapore demographic sensitivity** — Chinese/Malay/Indian is not just a research variable; it maps directly to Singapore's racial harmony legislation and the bank's internal diversity targets
- **Practical severity** — is a Cliff's δ of 0.667 "a big deal"? She needs a real-world translation
- **Candidate experience** — if candidates learn their CV was scored by an AI that showed ethnic or gender bias, what's the reputational risk?

---

## What They Will Challenge

1. "In plain English — does this mean the AI is racist? What would I say to a candidate who asked?"
2. "The study scored the same CV 240 times. But real CVs are all different. How do I know this applies to our actual candidate pool?"
3. "Gemini gave Indian male candidates a higher score. Does that mean it's favouring Indian men, or does it mean Indian men just happen to match the job description better in this template?"
4. "How much of a score difference is 'actionable'? If two candidates differ by 1 point, should I care?"
5. "My vendor says their system is 'bias-tested'. How do I know if they used something rigorous like this or just ticked a box?"
6. "What do I actually do with this? Do I stop using AI screening? Use a different model? Add a human review step?"
7. "Can I share this report with my CHRO? Is it written for that audience?"
8. "What's the difference between the Google model and the OpenAI model in terms of bias risk? Which one should I prefer?"

---

## What "Good" Looks Like

- A one-page lay summary — no statistics jargon, written for a non-technical HR professional
- A "What this means for hiring" section with concrete scenario: "If you use Gemini to pre-screen 100 CVs for a data analyst role in Singapore, approximately X% of [group] candidates would be disadvantaged relative to [other group]"
- A 'red / amber / green' summary per model: green = safe to use with human review, amber = use with caution, red = do not use
- Practical mitigation table: e.g., "Add a blind-review step", "Use human override for bottom 20%", "Audit quarterly"
- Language that connects to Singapore Fair Consideration Framework and any MAS guidelines on algorithmic fairness

---

## Dialect

Uses HR vocabulary: "candidate experience", "talent acquisition", "DEI targets", "unconscious bias", "shortlisting", "false rejection", "workforce fairness", "vendor assessment". Thinks in terms of process and policy. Wants to know what to DO, not just what the data shows. Will flag anything that sounds legally risky. Responds well to analogies and real-world scenarios.
