# Reading list — *When To and When Not to Loop*

Put downloaded PDFs in `papers/` (see the script at the bottom).
Priority: **[MUST]** read before finalising the paper · **[CORE]** directly relevant · **[BG]** background.

---

## 1. Must-reads — closest to this paper (check for overlap first)

| # | Paper | Why it matters for us | Link |
|---|---|---|---|
| 1 | **[MUST]** T. Ord, *Is there a half-life for the success rates of AI agents?* (2025) | Argues agent success decays exponentially with task length (constant hazard) — essentially our naive pⁿ model. Our Markov-with-recovery model predicts a *non-constant* hazard. Must be cited and positioned against. | https://arxiv.org/pdf/2505.05115 |
| 2 | **[MUST]** A. Sinha et al., *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs* (ICLR 2026) | Shows small per-step accuracy gains compound into large horizon gains, and reports *self-conditioning*: models get worse once their own errors are in context. Directly relevant to our reliability wall and coupling term. | https://arxiv.org/pdf/2509.09677 |
| 3 | **[MUST]** T. Kwa, B. West et al. (METR), *Measuring AI Ability to Complete Long Software Tasks* (2025) | The "50% time horizon" metric. Their data could be a second test of our reliability model. | https://arxiv.org/pdf/2503.14499 |
| 4 | **[MUST]** J. Kim et al., *The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective* (2025) | Systems view of agent cost, latency and energy — closest in spirit to our cost/latency walls. Check overlap carefully. | https://arxiv.org/pdf/2506.04301 |
| 5 | **[MUST]** S. Kapoor et al., *AI Agents That Matter* (2024) | Cost-aware agent evaluation; simple baselines competitive once cost counts. Already cited. | https://arxiv.org/pdf/2407.01502 |
| 6 | **[CORE]** Y. Kim et al., *Towards a Science of Scaling Agent Systems* (2025) | Quantitative study of when multi-agent setups help or hurt — the natural follow-up paper (multi-agent / USL). | https://arxiv.org/pdf/2512.08296 |

## 2. Surveys — to see what is happening around

| Paper | Focus | Link |
|---|---|---|
| L. Wang et al., *A Survey on Large Language Model based Autonomous Agents* (Front. Comput. Sci., 2025) | The standard agent survey (profile, memory, planning, action) | https://arxiv.org/pdf/2308.11432 |
| X. Yang, L. Li, H. Zhou, T. Zhu et al., *Toward Efficient Agents: A Survey of Memory, Tool Use, and Planning* (2026) | Efficiency of agents — closest survey to our topic | https://arxiv.org/pdf/2601.14192 |
| L. Mei et al., *A Survey of Context Engineering for Large Language Models* (2025) | Context management, compression, memory — our context wall and summarisation | https://arxiv.org/pdf/2507.13334 |
| A. Yehudai et al., *Survey on Evaluation of LLM-based Agents* (2025) | Agent benchmarks and evaluation | https://arxiv.org/pdf/2503.16416 |
| M. Mohammadi et al., *Evaluation and Benchmarking of LLM Agents: A Survey* (2025) | Evaluation taxonomy, incl. reliability and cost | https://arxiv.org/pdf/2507.21504 |
| M. Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (2025) | Failure taxonomy — useful for the silent vs detectable failure split | https://arxiv.org/pdf/2503.13657 |

## 3. Context length and degradation (the coupling term)

| Paper | Link |
|---|---|
| N. F. Liu et al., *Lost in the Middle* (TACL 2024) | https://arxiv.org/pdf/2307.03172 |
| C.-P. Hsieh et al., *RULER: What's the Real Context Size?* (COLM 2024) | https://arxiv.org/pdf/2404.06654 |
| K. Hong, A. Troynikov, J. Huber (Chroma), *Context Rot* (2025, tech report, not arXiv) | https://research.trychroma.com/context-rot |

## 4. Agent loops (the systems we are modelling)

| Paper | Link |
|---|---|
| S. Yao et al., *ReAct* (ICLR 2023) | https://arxiv.org/pdf/2210.03629 |
| N. Shinn et al., *Reflexion* (NeurIPS 2023) — recovery from errors | https://arxiv.org/pdf/2303.11366 |
| T. Schick et al., *Toolformer* (NeurIPS 2023) | https://arxiv.org/pdf/2302.04761 |
| S. Yao et al., *τ-bench* (2024) — pass^k reliability | https://arxiv.org/pdf/2406.12045 |
| C. E. Jimenez et al., *SWE-bench* (ICLR 2024) | https://arxiv.org/pdf/2310.06770 |
| C. Packer et al., *MemGPT* (2023) — summarisation/paging of memory | https://arxiv.org/pdf/2310.08560 |

## 5. Cost, routing, caching

| Paper | Link |
|---|---|
| L. Chen et al., *FrugalGPT* (2023) | https://arxiv.org/pdf/2305.05176 |
| I. Ong et al., *RouteLLM* (ICLR 2025) | https://arxiv.org/pdf/2406.18665 |
| I. Gim et al., *Prompt Cache* (MLSys 2024) | https://arxiv.org/pdf/2311.04934 |
| W. Kwon et al., *vLLM / PagedAttention* (SOSP 2023) | https://arxiv.org/pdf/2309.06180 |

## 6. "Loop" in the architecture sense (looped / recurrent-depth models)

Different meaning of "loop" (looping layers inside the model, not calling the model repeatedly), but the same when-to-stop question appears, and it connects to the LoGA / looped-ViT line.

| Paper | Link |
|---|---|
| M. Dehghani et al., *Universal Transformers* (ICLR 2019) — adaptive halting | https://arxiv.org/pdf/1807.03819 |
| A. Giannou et al., *Looped Transformers as Programmable Computers* (ICML 2023) | https://arxiv.org/pdf/2301.13196 |
| J. Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (2025) | https://arxiv.org/pdf/2502.05171 |
| F. Pappone, D. Crisostomi, E. Rodolà, *Two-Scale Latent Dynamics for Recurrent-Depth Transformers* (2025) | https://arxiv.org/pdf/2509.23314 |

## 7. Capacity-planning background (books — not freely downloadable)

- N. J. Gunther, *Guerrilla Capacity Planning* (Springer, 2007) — Universal Scalability Law.
- M. Harchol-Balter, *Performance Modeling and Design of Computer Systems* (Cambridge, 2013).

---

## Download script

Run from the repo root on your own machine (arXiv links only; the Chroma report and the books are not included).

```bash
mkdir -p papers && cd papers
while read -r name url; do
  [ -f "$name.pdf" ] || curl -sSL -A "loop-bounds-reading-list" -o "$name.pdf" "$url"
  sleep 3   # be polite to arXiv
done <<'EOF'
ord2025_halflife https://arxiv.org/pdf/2505.05115
sinha2025_illusion https://arxiv.org/pdf/2509.09677
metr2025_longtasks https://arxiv.org/pdf/2503.14499
kim2025_costdynamic https://arxiv.org/pdf/2506.04301
kapoor2024_agentsmatter https://arxiv.org/pdf/2407.01502
kim2025_scalingagents https://arxiv.org/pdf/2512.08296
wang2023_agentsurvey https://arxiv.org/pdf/2308.11432
efficientagents2026_survey https://arxiv.org/pdf/2601.14192
mei2025_contextengineering https://arxiv.org/pdf/2507.13334
yehudai2025_evalsurvey https://arxiv.org/pdf/2503.16416
mohammadi2025_evalsurvey https://arxiv.org/pdf/2507.21504
cemri2025_masfail https://arxiv.org/pdf/2503.13657
liu2024_lostmiddle https://arxiv.org/pdf/2307.03172
hsieh2024_ruler https://arxiv.org/pdf/2404.06654
yao2023_react https://arxiv.org/pdf/2210.03629
shinn2023_reflexion https://arxiv.org/pdf/2303.11366
schick2023_toolformer https://arxiv.org/pdf/2302.04761
yao2024_taubench https://arxiv.org/pdf/2406.12045
jimenez2024_swebench https://arxiv.org/pdf/2310.06770
packer2023_memgpt https://arxiv.org/pdf/2310.08560
chen2023_frugalgpt https://arxiv.org/pdf/2305.05176
ong2024_routellm https://arxiv.org/pdf/2406.18665
gim2024_promptcache https://arxiv.org/pdf/2311.04934
kwon2023_vllm https://arxiv.org/pdf/2309.06180
dehghani2019_universal https://arxiv.org/pdf/1807.03819
giannou2023_looped https://arxiv.org/pdf/2301.13196
geiping2025_recurrentdepth https://arxiv.org/pdf/2502.05171
twoscale2025_recurrentdepth https://arxiv.org/pdf/2509.23314
EOF
```

Note: all 28 arXiv PDFs were downloaded and checked against their first pages
(2026-11-03). Titles, authors and venues above were corrected from those pages
(METR title, Efficient-Agents survey title/authors, Two-Scale authors, venues for
Sinha, Wang and RouteLLM). PDFs are kept locally in `papers/` and are not
committed (arXiv licences generally do not permit redistribution).
