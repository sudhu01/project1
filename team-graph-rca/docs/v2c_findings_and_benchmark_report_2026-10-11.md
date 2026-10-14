# V2c findings and benchmark report

11 October 2026, IST

All 240 v2c physical captures are complete and verified. The full v2c experiment is unfinished because its first diagnostic check failed. With every evidence tool available, the fitted classifier correctly diagnosed 40 of 60 validation cases, or 66.7%. The required accuracy is 90%. The runner therefore stopped before learner training or reserved-test benchmarking.

The main finding is that a small increase in normal carts memory use caused the encoder to label ordinary cases as memory pressure. The physical memory fault was measurable, but the fitted category boundary did not remain reliable across capture sessions.

## What is complete

| Stage | Result |
|---|---|
| Physical collection | 240 accepted incidents, complete |
| Capture audit | Passed; unique physical episodes, public checksums and recovery verified |
| Data split | 120 training, 60 validation, 60 reserved test |
| Independent sessions | 40 total; 20 training, 10 validation, 10 test |
| Encoder and diagnostic model fitting | Complete, using training data |
| Full-evidence validation | Complete; 66.7%, failed the 90% gate |
| Expected SARSA and two-step SARSA training | Not started for v2c |
| Algorithm comparison on reserved test | Not run for v2c |
| Evidence sweep, cost sweep and synthetic challenges | Not run for v2c |

Across v2a, v2b and v2c, 600 physical incidents are now saved. The earlier 280 learner training runs and 2.8 million replay episodes still belong to v2a and v2b. V2c adds no learner training runs yet. Its planned learner seeds are 11, 22, 33, 44 and 55.

Physical capture means running a fault or healthy control in Docker and saving its measurements. Replay means reusing those saved observations. Model fitting learns how evidence relates to diagnoses. Policy training learns which evidence action to choose next. Completing capture does not complete those later stages.

## What v2c added

V2c contains six diagnosis classes: catalogue database stopped, orders stopped, payment CPU contention, orders-to-payment delay, carts memory pressure, and no fault. Each class has 20 training, 10 validation and 10 reserved-test incidents.

It adds carts memory and carts logs to the status, CPU, request-error, log and checkout-latency tools used earlier. Healthy scenarios rotate ordinary operation, higher load and temporary allocation followed by release. These controls help distinguish retained memory pressure from normal activity.

The protocol uses a 120-second healthy baseline, 30-second settling period, 120-second observation window and 10-second ingestion period. Collection runs one incident at a time.

## The measured diagnostic result

This check gives the classifier all evidence. It tests whether the measurements and model can distinguish causes before asking an algorithm to choose a limited set of probes. It is not a runnable policy constrained by the normal probe budget.

| Actual condition | Correct out of 10 | Accuracy | Prediction |
|---|---:|---:|---|
| Catalogue database stopped | 10 | 100% | Catalogue database |
| Orders stopped | 10 | 100% | Orders |
| Payment CPU contention | 10 | 100% | Payment |
| Orders-to-payment delay | 0 | 0% | Carts memory pressure |
| Carts memory pressure | 10 | 100% | Carts memory pressure |
| Healthy | 0 | 0% | Carts memory pressure |
| **Total** | **40 / 60** | **66.7%** | **Gate failed** |

These are validation results across ten sessions. Reserved-test responses have not been evaluated. The gate failure does not establish that SARSA, DP or another action-selection algorithm performs poorly on v2c; those comparisons have not run.

## Why it failed

The encoder turns memory measurements into labels such as NORMAL, PRESSURE and GROWTH. Its fitted carts pressure boundary was 57.734% of the container memory limit. A measurement above that boundary receives a pressure label.

| Measurement | Fraction of memory limit |
|---|---:|
| Fitted pressure boundary | 57.734% |
| Healthy training cases | 54.63% to 57.734% |
| Healthy validation cases | 57.79% to 58.22% |
| Payment-delay validation cases | 57.77% to 58.20% |
| Injected carts pressure, training | 83.46% to 83.63% |
| Injected carts pressure, validation | 83.54% to 83.59% |

Normal memory use moved slightly upward in later sessions. It stayed far below the injected fault, but crossed the category boundary. Every validation memory response became PRESSURE, including cases where carts was healthy. A strong signal in the raw numbers therefore became a misleading label.

Latency retained useful separation: all ten healthy validation cases had NORMAL latency, while all ten payment-delay cases had SLOW latency. All ten carts memory faults also had NORMAL latency. Memory pressure in this experiment did not automatically produce slow checkout.

Carts logs returned OTHER_ERROR in every training and validation case. That category did not distinguish the memory fault from other conditions. Adding a tool only helps diagnosis if its returned evidence separates the causes.

The tested smoothing values, 0.05, 0.1 and 0.2, all produced 66.7% validation accuracy. No smoothing change was applied. The investigation points to the memory category boundary and its sensitivity to normal drift, rather than a demonstrated benefit from changing smoothing.

## Algorithm benchmarks available now

There are no v2c measurements for policy accuracy, return, computation per action choice or average probes. The latest completed algorithm benchmark remains v2b. The table below repeats that benchmark for context; none of these numbers are v2c results.

Accuracy means correct diagnoses. Score is mean return after the probe-credit penalty. A probe is one evidence request. Decision time measures computation to choose an action, excluding physical tool waiting and training. Learner results average five seeds.

| Algorithm, v2b reserved test | Accuracy | Score | Average probes | Decision time, ms |
|---|---:|---:|---:|---:|
| Expected SARSA, tabular | 100% | 0.8400 | 1.800 | 0.0747 |
| Expected SARSA, hybrid | 100% | 0.8400 | 1.800 | 0.0915 |
| Two-step SARSA, tabular | 100% | 0.8400 | 1.800 | 0.0742 |
| Two-step SARSA, hybrid | 100% | 0.8400 | 1.800 | 0.0952 |
| DP planner | 100% | 0.8400 | 1.800 | 447.7385 |
| VOI planner | 100% | 0.8000 | 2.000 | 10.9371 |
| Selected fixed checklist | 100% | 0.8300 | 2.200 | 0.0028 |
| Dependency rule | 60% | 0.5100 | 1.800 | 0.0161 |
| Generalized rule | 60% | 0.4100 | 3.800 | 0.0027 |
| Random valid actions | 40% | 0.2716 | 2.788 | 0.0455 |
| Stop immediately | 20% | 0.2000 | 0 | 0.0010 |
| Defer immediately | 0% | 0.2500 | 0 | 0.0012 |

In v2b, both learners matched DP's accuracy, score and probe use with much less decision computation. The fixed checklist was faster to choose actions but used more probes. Neither learner passed promotion in the main v2b experiment because the return gain over the fixed baseline was 0.01, below the required 0.02.

Earlier qualifying gains remain limited to the declared v2b evidence subset without status and the high-cost setting. In the high-cost setting, Expected SARSA passed in all five seeds; two-step SARSA passed in three, while two seeds deferred. V2c provides no new evidence-sweep, cost-sweep, seed-stability or promotion result.

The previous report contains the full v2a tables and earlier sweep and challenge results: [benchmark report before v2c](pre_v2c_benchmark_report_2026-10-10.md).

## Reliability and limits

Interrupted captures were preserved and recovered. Failures included insufficient memory evidence coverage, degraded healthy controls, checkout recovery failure and a Docker timeout during CPU-helper cleanup. Rejected attempts were not counted as accepted incidents. Collection resumed after cleanup and health checks, with accepted captures and evidence gates retained.

The completion audit verified 240 distinct physical episodes, public bundle checksums, the declared split counts, unchanged runtime identity and no remaining lab lock or block. The original encoder, model and failed validation result remain saved. The older v1 reserved test was not reopened.

The 66.7% result applies to the current fitted classifier on this lab's validation cohort. It is not an algorithm ranking or a mathematical limit on what the raw measurements can support. A revised memory encoder may improve separation, but no such improvement has been verified. A revision needs its own recorded identity, training-only fitting and the unchanged diagnostic gate. The current failure must remain visible in any later report.

## Saved evidence

- `runs/adaptive_v2c/capture-completion-verification.json`: collection audit.
- `runs/adaptive_v2c/full-evidence-reference.json`: validation accuracy, confusion and smoothing checks.
- `runs/adaptive_v2c/diagnostic-gate-investigation.json`: training and validation memory categories and measured ranges.
- `runs/adaptive_v2c/phase-progress.json`: completed stages and the reason training stopped.
- `runs/adaptive_v2/pre-v2c-benchmark-data.json`: earlier verified algorithm benchmarks.

The next task is a separately recorded measurement revision that handles normal memory drift. Policy training and the remaining v2c benchmarks stay pending until the diagnostic gate passes.
