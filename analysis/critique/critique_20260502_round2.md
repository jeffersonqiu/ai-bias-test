# Round 2 Critique — Singapore LLM Demographic Bias Audit

**Date:** 2026-05-02 (Round 2)  
**Reviewer:** experiment-critique skill (manual execution)  
**Based on:** Round 1 critique `critique_20260502.md` + improvements described in prompt  
**Files reviewed:** `README.md`, `analysis/singapore_llm_bias_analysis.ipynb` (source cells), `bias_framework/stats.py`, `bias_framework/visualization.py`, `bias_framework/rubric.py`, `config/settings.py`

---

## Round 1 Critical Items — Resolution Checklist

| # | Item | Status | Notes |
|---|---|---|---|
| C1 | Bootstrap 95% CIs on Cliff's δ | **Resolved** | `cliffs_delta_ci()` added to `stats.py`; `run_pairwise_tests(bootstrap_ci=True)` called in §8.1; CI columns displayed as `95% CI lo` / `95% CI hi` in significant-pairs table and full pairwise table. The function correctly returns `(nan, nan)` for tied distributions (GPT). |
| C2 | GPT score-ceiling reclassification | **Resolved** | README table footnote is clear and accurate: "score-ceiling artefact / measurement null." Executive summary in notebook Cell 0 uses correct language ("Cliff's δ = 0.000 is a measurement null, not evidence of absent bias"). §8.1 narrative repeats the framing. |
| C3 | Power analysis | **Resolved** | §8.2 added with correct minimum-detectable-δ statement (≥0.47 at Bonferroni α) and the key practical implication about GPT unmeasurability. |
| C4 | Harm taxonomy | **Resolved** | §10A added with four scenarios, Likelihood × Severity risk scores, reversibility ratings, and applicable regulations (Singapore FCF, EU AI Act, MAS TRM). Risk scores are computed and colour-coded. |
| C5 | Executive summary | **Resolved** | Cell 0 now opens with a full "Executive Summary" section: three key findings, recommended action, and clear Phase 2 caveat. The recommended action is specific ("Do not use Gemini 2.5 Flash Lite for CV scoring without demographic-blind prompting or post-hoc score normalisation"). |
| C6 | DBS Bank anonymisation | **Resolved** | `job_posting.py` now reads `ORGANISATION: Singapore Retail Bank (anonymised)`. Confirmed no remaining references to "DBS Bank Ltd" or "Marina Bay Financial Centre". |
| C7 | HR Professionals section in README | **Resolved** | Full "For HR Professionals" section added to README with plain-language narrative, practical implications (hiring funnel scenario with specific candidate-group framing), four actionable recommendations, and a RAG model rating table. |

All 7 Critical items from Round 1 are resolved.

---

## Round 1 Major Items — Resolution Checklist

| # | Item | Status | Notes |
|---|---|---|---|
| M8 | Intersectional decomposition | **Resolved** | §9.1 computes per-cell interaction residuals (cell mean minus ethnic + gender main effects + grand mean). Displayed per model with colour-gradient table. §10.1 Interpretation explicitly calls out the intersectional nature. |
| M9 | Mitigation options | **Resolved** | §10B added with 5 mitigations (name redaction, model substitution, score normalisation, human-in-the-loop, periodic re-audit), each with Effort, Effectiveness, and Tradeoff columns. |
| M10 | `score_distributions` in `visualization.py` + `generate_all()` | **Resolved** | `score_distributions()` now in `visualization.py`; included as first call in `generate_all()`; notebook cell uses `viz.score_distributions(df, CHARTS_DIR)`; `CONDITION_LABELS`/`CONDITION_COLORS` imported from module instead of redefined inline. |
| M11 | Reasoning-text analysis expanded | **Resolved (partial)** | §11 now iterates all 6 conditions × all models, showing one representative sample per cell (12 samples total). However, no keyword search or systematic cross-cell comparison is performed — the analysis is still display-only with explanatory notes. The RAI Officer finding about keyword frequency analysis is deferred to Phase 2. |
| M12 | Rubric verbatim in notebook | **Resolved** | §3 includes the heading "The verbatim system prompt sent to every model is reproduced below" and the subsequent code cell prints `SYSTEM_PROMPT` to the executed output. The prompt text appears in notebook output on execution, making the audit self-contained. |
| M13 | Decision gate in README | **Not resolved** | README Phase 2 section still has no "Phase 1 is sufficient to recommend against Gemini" paragraph and no formal governance gate for when Phase 2 results may inform hiring decisions. |
| M14 | `worst_ethnic_pair_d` column | **Resolved** | Added to `compute_bias_summary()`; displayed in notebook §9 bias summary table; reported in §10 conclusions code cell. |
| M15 | RAG model safety rating in notebook | **Not resolved** | RAG rating exists in README "For HR Professionals" section, but there is no equivalent in the notebook itself. The §10B mitigation table and §10A harm taxonomy serve a related purpose, but the notebook lacks a per-model verdict cell that a stakeholder reviewing only the notebook would see. |
| M16 | Hiring funnel worked example in notebook | **Partially resolved** | The README "For HR Professionals" section contains a concrete hiring-funnel scenario ("100 CVs… four of the six demographic groups would receive a lower baseline score"). The §10A harm taxonomy scenario table covers the use-case framing. However, the notebook itself has no quantified hiring-funnel example; this is covered only in the README. |
| M17 | Explicit model recommendation paragraph in notebook | **Partially resolved** | The Executive Summary in Cell 0 provides a recommended action ("Do not use Gemini 2.5 Flash Lite…"). However, §10 Conclusions has no explicit "therefore" paragraph stating which model a product team should choose for production. The §10.1 Interpretation section describes the finding accurately but stops short of a product-decision sentence. |

5 of 10 Major items fully resolved, 3 partially resolved, 2 not resolved.

---

## Remaining Issues

| Severity | Persona | Finding | Suggested Fix |
|---|---|---|---|
| **Major** | Product Lead | **No decision gate in README Phase 2 section.** The README Phase 2 instructions contain no statement that Phase 1 is sufficient to exclude Gemini 2.5 Flash Lite from production use without mitigation. A product lead evaluating vendor selection cannot tell whether they need to wait for Phase 2 or whether Phase 1 alone warrants action. | Add 1 paragraph to README "## Phase 2" section: "Phase 1 alone is sufficient to recommend against deploying `gemini-2.5-flash-lite` for name-sensitive CV scoring in production without mitigation (large δ, 8 significant pairs). Phase 2 will determine whether the bias is specific to the flash-lite tier or present across the Gemini 2.5 family." |
| **Major** | HR Business User, Product Lead | **No RAG model rating in notebook.** The README has a clear RAG table (Red/Amber/Green), but a stakeholder viewing only the executed notebook PDF encounters no equivalent signal. The notebook's §10 Conclusions has statistics and limitations, but no per-model verdict in lay language. | Add a small markdown table to §10.1 or the Executive Summary cell: `| Model | RAG Rating | Verdict |` — matching the README RAG table. This makes the notebook fully self-contained for non-technical reviewers. |
| **Major** | Product Lead | **No plain-language translation of Cliff's δ anywhere in the notebook.** The Executive Summary says "pooled Cliff's δ = 0.667" but never translates this to a percentage-of-trials statement. The original Round 1 critique identified this as a Critical gap; it was resolved in the README (HR section describes "Indian Male and Malay Female names received a score of 9; all other names received a score of 8") but the notebook never converts δ to a human-readable probability. For Gemini, δ = 0.667 means the favoured condition scored higher in 83% of head-to-head trial pairs. | Add a sentence to the Executive Summary cell or §10.1: "In practical terms: in pairwise comparisons between a score-9 condition and a score-8 condition, the score-9 condition ranked higher in 83% of trial pairs (Cliff's δ = 0.667)." |
| **Major** | RAI Officer | **No significance test on intersectional residuals.** §9.1 computes interaction residuals numerically and displays them with colour coding, but does not test whether the Indian×Male and Malay×Female residuals are statistically non-zero. The Round 1 RAI critique specifically requested this test. With n=20 per cell and a bimodal distribution, a bootstrap permutation test or a simple comparison of the residual against the null (residual = 0) would elevate this from visual to inferential. | Add a bootstrap permutation test of the largest residual against a null distribution (shuffle condition labels within model, recompute residuals, report p-value). Even a brief sentence noting "the Indian Male and Malay Female residuals are [significant / not significant] under permutation (p = X)" would satisfy this requirement. |
| **Major** | Data Scientist | **README Limitations section still says "No confidence intervals."** Limitation #5 reads: "No confidence intervals — Cliff's δ is reported as a point estimate; bootstrap CIs would quantify uncertainty." This is now incorrect — bootstrap CIs have been implemented in `stats.py` and are shown in the notebook §8.1 pairwise table. The Achievements section still shows "🔲 Bootstrap confidence intervals on Cliff's δ" as an uncompleted item. Both references are stale and will confuse anyone comparing README to notebook. | Update README Limitations #5 to "Bootstrap 95% CIs on pairwise Cliff's δ are reported in §8.1 of the analysis notebook. CIs on the pooled ethnic/gender summary δ are not yet propagated to `compute_bias_summary()`." Update Achievements to "✅ Bootstrap 95% CIs on Cliff's δ (pairwise)." |
| **Minor** | Data Scientist | **No git commit hash in reproducibility appendix.** The Round 1 Minor finding (item M19) remains unaddressed. The reproducibility cell captures package versions and execution timestamp but does not capture the git commit hash. The cell notes "Model IDs refer to API-resolved versions as of execution date" but does not actually capture the resolved version metadata from the raw API response. | Add `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()` to the reproducibility cell. Note that `raw_response` API version capture requires per-provider implementation effort and can remain a Phase 2 item. |
| **Minor** | Data Scientist | **Reasoning-text analysis is display-only; no keyword search or semantic comparison.** §11 shows one sample per cell (12 samples) but notes that keyword frequency and semantic similarity analysis is "left as a Phase 2 enhancement." The Round 1 RAI critique asked specifically whether Gemini's reasoning for score-9 conditions (Indian Male, Malay Female) uses markedly different language. This is answerable with the existing CSV data (simple string comparison of the 2 score-9 samples vs. 4 score-8 samples) and requires no additional API calls. | Add a cell in §11 comparing the 2 score-9 reasoning strings against the 4 score-8 reasoning strings: at minimum, check whether they are identical or whether the score-9 reasoning contains any additional language, name references, or qualitatively different framing. Even noting "the reasoning text is character-for-character identical across all conditions" (which would be striking) resolves the gap. |
| **Minor** | Product Lead | **Dimension heatmap narrative does not identify which dimension drives the Gemini score split.** §7 says "column-wise variation within a single model panel indicates dimension-specific bias" but no follow-up cell explains whether the 8-vs-9 split in Gemini is uniform across all five sub-scores or concentrated in one dimension. This interpretation should appear as a brief narrative cell after the chart. | Add a markdown cell after the dimension heatmap: "For Gemini 2.5 Flash Lite, the 8-vs-9 holistic score split is [uniform across all 5 dimensions / concentrated in X dimension], suggesting the bias originates in [holistic scoring / specific dimension assessment]." This can be confirmed by inspecting the chart output. |
| **Minor** | RAI Officer | **Notebook §10.2 Limitations table still lists "No bootstrap CIs on pooled δ" as a gap.** The pairwise CIs are implemented, and the pooled summary CIs are indeed still missing — but the limitation text does not distinguish pairwise vs. pooled, leaving the reader uncertain whether any CIs exist at all. | Revise §10.2 limitation row to: "Bootstrap CIs on pooled ethnic/gender summary δ in `compute_bias_summary()` not yet propagated; pairwise CIs are computed and shown in §8.1." |

---

## New Issues Introduced by Round 1 Changes

| Severity | Issue | Detail |
|---|---|---|
| Minor | README Limitations #5 and Achievements section are now inconsistent with the implementation. CIs exist in the notebook but README still shows them as missing. (Detailed above as a remaining issue.) | |
| Minor | The reasoning-text analysis (§11) is new and technically correct, but the note "Full reasoning-text analysis is left as a Phase 2 enhancement" may give a false impression that nothing was done. The section heading and the note should be reconciled: either rename the section to "§11 Reasoning Text Samples (Phase 2: keyword analysis)" or add a brief observation about whether the Gemini score-9 reasoning is distinguishable from score-8 reasoning. | |

---

## Overall Verdict

**Minor cleanup remaining.**

All 7 Critical items from Round 1 are resolved, and 8 of 10 Major items are either fully or partially resolved. The two unresolved Majors (decision gate in README, RAG rating in notebook) are low-effort fixes (each is a single table or paragraph). The remaining issues are concentrated in three clusters: (1) stale README text that now contradicts the implementation, (2) the notebook lacking a few lay-audience signals that exist only in the README, and (3) minor statistical gaps in the intersectional and reasoning-text analyses. None of these block internal or peer-group circulation. The document is not yet ready for public release or regulatory submission (the significance test on intersectional residuals and the decision-gate language should be added first), but it is suitable for wider internal review and client-facing discussion with the caveat that Phase 2 is pending.

---

*Report generated 2026-05-02 by manual review (experiment-critique skill unavailable in this environment).*
