# Benchmark report before v2c

10 October 2026, IST

V2a and v2b are complete. Their main comparisons, evidence sweeps, cost sweeps and synthetic challenges are saved and verified. V2c has not started. The strongest current practical result is that learned policies match DP on the main v2b test with much less decision computation. Expected SARSA also behaves more consistently than two-step SARSA when probe costs are high.

## What has been completed

| Item | V2a | V2b |
|---|---:|---:|
| Accepted physical incidents | 160 | 200 |
| Training / validation / reserved test | 80 / 40 / 40 | 100 / 50 / 50 |
| Independent capture sessions | 40 | 40 |
| Evidence combinations | 7 | 9 |
| Probe-cost settings | 5 | 5 |
| Main learner training runs | 10 | 10 |
| Evidence-sweep learner runs | 70 | 90 |
| Cost-sweep learner runs | 50 | 50 |

Every learner run uses 10,000 replay training episodes. Together that is 280 declared training runs and 2.8 million replay episodes. These are repeated training experiences from saved captures, not 2.8 million new physical incidents. Each learner uses seeds 11, 22, 33, 44 and 55. A seed changes training randomness.

V2a covers catalogue-db, orders, payment and no fault, with status, CPU, request-error and log evidence. V2b adds checkout latency and the orders-to-payment delay path. Memory measurement and its retained-allocation helper have engineering readiness checks, but their expanded physical benchmark belongs to v2c. There are no completed v2c algorithm results yet.

Accepted capture receipts, session-based data splits, train-only fitting, source/model/policy identities, checkpoints and measured recovery checks were retained. Disk-full failures and interrupted work were preserved and recovered without changing the evidence gates. The older v1 reserved test was not reopened.

## Reading the numbers

Accuracy is the fraction of incidents diagnosed correctly. Score is the saved mean return: a correct answer earns 1, a wrong answer 0, and deferral 0.25, with probe credits penalized by the configured cost weight. A probe is one evidence request. Credits represent the assigned price of that request, so fewer probes do not always mean lower cost.

Decision time measures computation needed to choose an action. It excludes physical tool waiting time and training time. It is not the total time to resolve an incident. Learner and random-policy table entries average the five seeds. Tabular policies use learned state/action values; hybrid policies can use a planner fallback.

## Main v2b benchmark

The reserved test contains 50 incidents in ten independent sessions. Policies and the fixed baseline were selected using validation before this test.

| Algorithm | Accuracy | Score | Probes | Credits | Decision ms |
|---|---:|---:|---:|---:|---:|
| Expected SARSA, tabular | 100.0% | 0.8400 | 1.800 | 3.200 | 0.0747 |
| Expected SARSA, hybrid | 100.0% | 0.8400 | 1.800 | 3.200 | 0.0915 |
| Two-step SARSA, tabular | 100.0% | 0.8400 | 1.800 | 3.200 | 0.0742 |
| Two-step SARSA, hybrid | 100.0% | 0.8400 | 1.800 | 3.200 | 0.0952 |
| DP planner | 100.0% | 0.8400 | 1.800 | 3.200 | 447.7385 |
| VOI planner | 100.0% | 0.8000 | 2.000 | 4.000 | 10.9371 |
| Selected fixed checklist | 100.0% | 0.8300 | 2.200 | 3.400 | 0.0028 |
| Dependency rule | 60.0% | 0.5100 | 1.800 | 1.800 | 0.0161 |
| Generalized rule | 60.0% | 0.4100 | 3.800 | 3.800 | 0.0027 |
| Stop immediately | 20.0% | 0.2000 | 0.000 | 0.000 | 0.0010 |
| Defer immediately | 0.0% | 0.2500 | 0.000 | 0.000 | 0.0012 |
| Random valid actions | 40.0% | 0.2716 | 2.788 | 4.208 | 0.0455 |

## Main v2a benchmark

The reserved test contains 40 incidents in ten independent sessions. Policies and the fixed baseline were selected using validation before this test.

| Algorithm | Accuracy | Score | Probes | Credits | Decision ms |
|---|---:|---:|---:|---:|---:|
| Expected SARSA, tabular | 100.0% | 0.8625 | 2.250 | 2.750 | 0.0818 |
| Expected SARSA, hybrid | 100.0% | 0.8625 | 2.250 | 2.750 | 0.1113 |
| Two-step SARSA, tabular | 100.0% | 0.8625 | 2.250 | 2.750 | 0.0823 |
| Two-step SARSA, hybrid | 100.0% | 0.8625 | 2.250 | 2.750 | 0.0994 |
| DP planner | 100.0% | 0.8500 | 1.500 | 3.000 | 337.9577 |
| VOI planner | 100.0% | 0.8500 | 1.500 | 3.000 | 8.2642 |
| Selected fixed checklist | 100.0% | 0.8625 | 2.250 | 2.750 | 0.0028 |
| Dependency rule | 75.0% | 0.6625 | 1.750 | 1.750 | 0.0161 |
| Generalized rule | 75.0% | 0.5875 | 3.250 | 3.250 | 0.0030 |
| Stop immediately | 25.0% | 0.2500 | 0.000 | 0.000 | 0.0010 |
| Defer immediately | 0.0% | 0.2500 | 0.000 | 0.000 | 0.0008 |
| Random valid actions | 41.0% | 0.3058 | 2.625 | 3.835 | 0.0438 |

Expected SARSA and two-step SARSA reached the same accuracy, score and probe use in all five seeds of each main experiment. Their small decision-time difference does not establish a winner. Hybrid mode did not improve the main scores and added some computation.

On v2b, tabular learners take about 0.075 ms per decision, versus 448 ms for DP and 10.94 ms for VOI. That is roughly 6,000 times less decision computation than DP in this saved run. The fixed checklist is faster still, about 0.003 ms, but uses 2.2 probes versus 1.8 for the learners. Rules are cheap to run but miss 40% of v2b cases. Deferral has zero diagnosis accuracy because it gives no diagnosis; its 0.25 score is the explicit deferral reward.

V2a and v2b have different fault scopes. V2b adds a fifth class and a path fault, so their scores should not be treated as a controlled before/after comparison. In v2a, learned policies matched the fixed checklist; they did not establish a performance improvement.

## What the evidence tests show

Each row reports the policy selected on validation for that evidence subset, evaluated on the same reserved v2b cohort. It is not a newly collected cohort.

| Available evidence | Validation full-evidence accuracy | Selected test accuracy | Score | Probes |
|---|---:|---:|---:|---:|
| status | 60% | 60% | 0.51 | 1.8 |
| cpu, status | 80% | 80% | 0.65 | 2.4 |
| cpu, request_errors, status | 80% | 80% | 0.65 | 2.4 |
| cpu, logs, request_errors, status | 80% | 80% | 0.65 | 2.4 |
| cpu, latency, logs, request_errors, status | 98% | 100% | 0.84 | 1.8 |
| cpu, latency, logs, request_errors | 98% | 100% | 0.82 | 1.8 |
| latency, logs, request_errors | 78% | 80% | 0.66 | 1.4 |
| latency, logs | 58% | 60% | 0.50 | 1.0 |
| latency | 58% | 60% | 0.50 | 1.0 |

CPU and latency together distinguish the expanded faults well. Latency alone is not enough; logs and request errors cannot fully replace the missing CPU evidence in this scope. Seven subsets missed the declared diagnostic gate. Their training completed for comparison, but completion does not mean that gate passed.

The v2a sweep showed the same general pattern in its smaller scope: status alone reached 75% test accuracy, adding CPU reached 100%, logs plus request errors reached 75%, and logs alone reached 50%. No v2a evidence subset produced a qualifying RL promotion.

## Cost tests and seed stability

The cost weight controls how strongly the score penalizes probe credits. These sweeps retrain each declared setting separately.

| Cost weight | V2a selected score / probes | V2b selected score / probes | V2b selected accuracy |
|---|---:|---:|---:|
| 0 | 1.0000 / 2.25 | 1.00 / 2.2 | 100% |
| 0.025 | 0.9313 / 2.25 | 0.92 / 1.8 | 100% |
| 0.05 | 0.8625 / 2.25 | 0.84 / 1.8 | 100% |
| 0.1 | 0.7250 / 2.25 | 0.68 / 1.8 | 100% |
| 0.2 | 0.4500 / 2.25 | 0.36 / 1.8 | 100% |

At v2b cost weight 0.2, all five Expected SARSA seeds achieved 100% accuracy, score 0.36 and 1.8 probes. Two-step SARSA seeds 33, 44 and 55 matched that result; seeds 11 and 22 deferred every case, scoring 0.25 with zero probes. Thus its five-seed mean diagnosis accuracy is 60%, score 0.316 and 1.08 probes. The low probe count reflects deferral, not more efficient successful diagnosis. This happened in both tabular and hybrid forms.

## Where improvement was demonstrated

- Main v2a: no RL promotion. The learned policies matched the selected fixed baseline.
- Main v2b: no RL promotion. Learners scored 0.84 versus 0.83 for the selected fixed baseline. The gain of 0.01 is below the declared 0.02 requirement.
- V2b subset with CPU, latency, logs and request errors, without status: every seed of both learners passed. Score was 0.82 versus 0.80 for the selected fixed baseline, with 1.8 versus 2.0 probes and 100% accuracy. The saved paired session interval was positive.
- V2b high-cost setting 0.2: all Expected SARSA seeds and three of five two-step seeds passed. Their score was 0.36 versus fixed baseline 0.32. The two deferring seeds did not pass.

Promotion requires the diagnostic gate, at least 90% accuracy, at least 0.02 return advantage over the validation-selected fixed baseline, and a positive paired interval. A promotion against the fixed baseline does not establish superiority over DP: DP matches the successful learners in these settings. Gains belong to the declared settings and do not justify a claim that RL always wins.

## Synthetic outage and price challenges

These perturb saved evidence or prices. They add zero physical incidents. The table tracks each main experiment's validation-selected policy, fixed checklist for v2a and DP for v2b.

| Challenge | V2a accuracy / score | V2b accuracy / score | V2b probes |
|---|---:|---:|---:|
| all-telemetry-outage | 75% / 0.5125 | 60% / 0.21 | 4.8 |
| double-prices | 100% / 0.7250 | 100% / 0.68 | 1.8 |
| half-prices | 100% / 0.8875 | 100% / 0.91 | 1.8 |
| logs-outage | 100% / 0.8625 | 100% / 0.84 | 1.8 |
| request-metrics-outage | 100% / 0.8625 | 80% / 0.42 | 4.0 |
| stats-outage | 75% / 0.5125 | 80% / 0.64 | 1.8 |

V2b tolerates lost logs, but losing stats or request metrics reduces accuracy to 80%. Losing all telemetry reduces it to 60% and increases probe use. Good normal-test accuracy therefore does not guarantee reliable diagnosis during an outage.

## Limits and next phase

The main reserved cohorts contain 40 and 50 incidents, with ten independent sessions each. Algorithms and seeds reuse those incidents; they are not independent physical replications. Perfect test accuracy is a result for these measured lab cases, not a guarantee for an unfamiliar production system. Timings depend on this machine and its load. Models and policies were fitted or selected without using reserved test data.

V2c will add carts memory pressure and healthy allocation controls, collect its own measured cohort, then repeat the same verification and benchmarks. Memory results remain pending. The pre-v2c report is now available; lab identity, cleanup and health checks must pass before collection starts.

Evidence for this report is saved in runs/adaptive_v2a and runs/adaptive_v2b, including capture audits, main comparison audits, evidence-sweep-verification.json, cost-sweep-verification.json and main challenge verification. Extracted benchmark values are in runs/adaptive_v2/pre-v2c-benchmark-data.json.
