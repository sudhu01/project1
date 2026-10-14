# Adaptive v2 implementation

Started on 7 October 2026. Owner for every step is the RCA implementation maintainer.
Branch `codex/adaptive-evidence-v2`. New reports belong in `runs/adaptive_v2`.

## Reference and current implementation

The starting source revision is `aa346b76092c81df548633674401c92dda31d86c`.
`experiments/stop_v1_result.json` records 120 evaluation captures, the frozen
selection and report checksums. Its reserved test remains closed. V1 records,
datasets, fitted models and checkpoints must retain their original encoding.

The active package is `src/rca_investigation`, separate from the parent simulator
and legacy `pipeline` application. At the starting revision, config validation permits only
status actions for v1. Capture accepts two stopped resources and healthy controls.
Docker status snapshots are current-window observations; CPU uses process metrics
in the Prometheus adapter. Existing checkout readiness does not establish a
repeated payment workload. Rules hard-code two resources; Q keys omit entry
resource and action mask. Objective propagation needs revision before new sweeps.

## Steps and acceptance checks

| Step | Work and source | Acceptance check | Status |
|---|---|---|---|
| 1 | Checklist and reference artifacts, this file | Every action has a contract, source and verification record | Complete |
| 2 | Contracts, config, catalogue, dataset, experiment | Stable v1 hashes; strict v2 tools, packets and causes | Passed |
| 3 | Capability inventory, `configs/resources_sockshop_v2.yaml` | Exact live identities, measured limits, verified selectors and finite prices | Passed |
| 4 | Raw store, encoding, composite backend | Immutable reproducible derivation; only purchased evidence revealed | Passed |
| 5 | Container stats and CPU metrics | Exact units, coverage and quotas; measured contention and checkout effect | Passed engineering confirmation |
| 6 | Request metrics | Matching denominator; no traffic unavailable; transport failures separate | Passed software and live checkout checks |
| 7 | Log parser and vocabulary | Bounded observed peer references; verified ingestion; controller logs excluded | Passed software and ingestion checks; mappings fitted from training only |
| 8 | Latency metrics | Operation-specific timing, adequate samples, verified path | Passed engineering pilot |
| 9 | Memory metrics | Consistent bytes and limit, pressure/growth, allocation control | Passed resource-pressure pilot |
| 10 | Workload and capture | Repeated checkout; correct collection window; recovery excluded | Passed mechanics pilot and cart operations; evaluation capture running |
| 11 | Fault handlers | Independently measured effects and verified cleanup for each mechanism | Passed engineering pilots; lease helper images built separately |
| 12 | Encoder and model | Frozen session splits; training-only fitting; distinguishable labels | Software pipeline passed; physical cohort collecting |
| 13 | RL state and training | Public version/context/mask key; common utility; controlled fixtures learned | Passed; twenty fixture runs completed across both learners and five seeds |
| 14 | Comparison and crossover | Fair baselines, pure/hybrid separation, paired uncertainty | Implemented; physical evaluation pending |
| 15 | CLI and regression checks | Profile paths, leakage checks, resume through CLI | Passed; all three profiles validate; latest full regression: 173 passed |
| 16 | Promotion | Physical evidence before capture/training/performance claims | Pending |

## Action verification register

All actions use the v2 tool and packet contracts in `contracts_v2.py`.
Physical engineering pilots passed. Five accepted v2a mechanics captures remain
separate from the independently assigned evaluation cohort. Physical policy
evaluation has not completed.

| Family | Adapter source planned | Contract | Adapter test | Physical artifact |
|---|---|---|---|---|
| status | `backends/docker.py` | Step 2 | Contract/replay tests | `runs/adaptive_v2/capabilities.json` |
| cpu | `backends/container_stats.py`, `metrics/cpu.py` | Step 2 | Quota, coverage, bounds tests | `runs/adaptive_v2/cpu-pilot-tail-confirmation.json` |
| request_errors | `backends/request_metrics.py`, `metrics/requests.py` | Step 2 | HTTP/transport denominators, no traffic | `runs/adaptive_v2/checkout-verification.json` |
| logs | `logs/parser.py` | Step 2 | Exact peer, aliases, bounds, controller exclusion | `runs/adaptive_v2/capabilities.json` |
| latency | `metrics/latency.py` | Step 2 | Successful-only p50/p95, twenty samples | `runs/adaptive_v2/latency-pilot.json` |
| memory | `backends/container_stats.py`, `metrics/memory.py` | Step 2 | Byte parsing, coverage, robust sustained growth | `runs/adaptive_v2/memory-pilot.json` |

Offline reports establish software behavior. Engineering pilots establish signal
and recovery behavior. New reserved cohorts establish physical policy results.

## Findings and limits

The original two-core CPU trial did not establish the required checkout effect.
A quarter-core trial with the standard scheduler period also missed the median
criterion. Those failed reports remain saved. A separate confirmation used a
quarter-core quota with a one-second scheduler period and a declared p95 criterion.
It passed sustained contention, a checkout tail increase and recovery. The
evaluation uses that recorded quota and verifies the effect for every capture.

Memory allocation raised measured usage and released cleanly. Checkout stayed
healthy. The carts profile therefore diagnoses retained resource pressure; it
does not establish a leak or customer degradation.

The first mechanics capture exposed Docker's zero-limit row for a stopped
container. That row now becomes unavailable evidence. The rejected attempt
remains in the audit directory; the five later mechanics captures passed.

The common evaluation protocol is baseline 120 seconds, settling 30 seconds,
observation 120 seconds and telemetry ingestion allowance 10 seconds. Captures
use one dedicated workload session. Cart operations are separately timed in v2c.
No recovery samples enter the observation window.

The evaluation schedules declare twenty training, ten validation and ten reserved
test incidents per cause, grouped into independent named sessions. Healthy
controls rotate between ordinary and higher load; v2c adds allocate-and-release.
Engineering pilots are never fitted as evaluation data.

Checkout latency uses successful responses only. The encoder fits the 95th
percentile of healthy training windows' p95 values, using linear interpolation
at index `(n - 1) * p`. Windows need at least twenty successes. This matches
the statistic used by the latency packet. Memory growth uses the median of
long-span pair slopes, with at least ninety seconds of observation.

The fixed baseline searches at most 32 orders and five stopping thresholds on
validation. It includes orders based on individual information and on conditional
information from the fitted joint prototypes. Useful prefix permutations rotate
their first action, so the search limit retains different starting probes.
Comparisons retain the v1 rule transfer, a generalized rule, one/two
step planning, deeper planning on small catalogues, both pure learners and their
separate hybrids. Paired intervals resample capture sessions. Mixture sweeps keep
methods and priors frozen. Source-outage challenges remain synthetic.

The latest full software regression passed 173 tests in 213.87 seconds. It includes
the actual CLI resume checks for both learners, the raw-to-model pipeline,
the validation gate identity, the fixed baseline's correlated-evidence case,
immutable source snapshots, failed-capture audit retention,
and unchanged v1 artifact hashes. Status packets
retain the actual snapshot time; aggregate measurements use the window end.
The final model-write review also passed fifteen focused checks. A v2 fitted
model can be reused with identical inputs, but changed inputs require a new
artifact path.

Source and artifact identities freeze before physical replay training and final
selection. Evidence subsets and cost settings preserve all causes, share the raw
cohort, and receive separate derived packets, fitted models and checkpoints.
Each experiment also saves an immutable package-source snapshot before fitting.
Its runtime hash matches the training identity. A changed source tree or altered
snapshot requires a new experiment path. The snapshot preserves the actual code
behind earlier results when later profiles need a fix.

The active sequence runner waits for v2a, then fits, trains and evaluates it.
Before v2b it installs the pinned lease helper images and repeats the engineering
checks, preserving earlier reports. It runs v2b and v2c in order. Failed evidence
gates remain recorded for weak subsets and prevent performance promotion.
Physical policy results remain pending until those runs finish.

The hourly thread follow-up `Complete adaptive RCA phases` checks the active
runner, fixes failures within the existing task, and reports completed phases.
Normal collection progress stays quiet. It must stop scheduling when the plan
finishes or a major blocker needs the user's action. Use
`runs/adaptive_v2/follow-up-state.json` to track physical reports already sent.

A collection-time audit checked the first 44 accepted training captures without
opening validation or reserved test data. Public measurements and recovery
receipts matched their checksums. Every running resource had sufficient stats
coverage, and all 220 log collections succeeded. Stopped resources returned
unavailable stats. The audit is saved in
`runs/adaptive_v2/training-capture-health.json`; it does not fit thresholds or
replace the full-cohort validation gate.

The active v2a controller was started before the final lease and audit-status
changes. It retains the earlier bounded helper binary and pin for this cohort.
Published raw receipts and accepted schedule slots establish its terminal
status; its older attempt snapshots may still say `CREATED` after recovery.
Later cohorts use the lease helper and save the final attempt status explicitly.
The running cohort's code and helper are kept constant until collection ends.

Later collectors also retain rejected observation windows in a separate audit
directory, with a checksum and bounded failure message. Recovery failures record
their own reason. These observations never enter the accepted dataset. If a
failed observation exceeds eight megabytes, its audit retains the window, size
and checksum rather than copying the oversized measurements. Failed baseline
requests are saved before their acceptance check, so a baseline rejection records
the evidence without injecting a fault. The regression checks recovery,
rejection, isolation and bounds.

On October 7, v2a stopped after 73 accepted schedule slots when Windows denied
the final directory rename for the next completed training capture. Its effect
and recovery checks had passed. Recovery verified all four saved checksums,
published the staged directory without changing bytes, and restored slot
`v2a-train-19/1`. The schedule now contains 74 accepted incidents. The original
failure reports and schedule snapshot remain in
`runs/adaptive_v2a/publication-recovery/`, with the recovered raw hash.

Both interrupted controllers had exited before recovery. All lab containers were
running, and five semantic checkouts passed before collection resumed.
`scripts/resume_v2a_legacy.py` resumes this cohort with the original helper hash,
210-second CPU lifetime and eight workers. It permits only that dispatch and
requires controller ownership and an executable check for cleanup. Container
identities, helper pins, accepted observations and the schedule remain unchanged.

The resumed collector uses the current audit and semantic-checkout validation.
This is a collector revision within the v2a cohort and must remain visible when
interpreting its results. It changes no thresholds or fault settings. Publication
now retries a transient sharing lock for five seconds and retains the staged
capture if the lock persists. Eleven focused publication, helper and raw-capture
tests passed. The phase sequence was restarted to wait for v2a and then continue
the existing v2b and v2c plan.

At the next hourly check, collection had stopped at 78 accepted slots. Orders
started its HTTP server at 15:50:29 UTC, about 142 seconds after its recorded
restart, exceeding the unchanged 90-second capture recovery limit. Incident
`c162c208-e134-4c31-a676-4cd3348c0ee4` remains `RECOVERY_FAILED` and outside the
accepted dataset. Its rejected observations and attempt receipt remain saved.
The failed capture, phase report, schedule snapshot and successful manual
recovery report are preserved under `runs/adaptive_v2a/recovery-retries/`.

Both controllers had exited before manual recovery. Starting the same orders
container and verifying five healthy checkouts restored the lab without replacing
containers. The v2a resume wrapper now handles this specific recovery failure
with separate attempt reports. It preserves the failed attempt, confirms manual
recovery, and collects a new incident for the unfinished slot. Three consecutive
failures without accepted progress stop the wrapper and report failure. The
phase runner sees the final failure only after bounded recovery attempts end.
The wrapper now also propagates the CLI exit code. Thirteen focused checks
passed, including failed-attempt isolation and the retry limit. The 90-second
acceptance limit, effect gates and original helper pin remain unchanged.
Collection and the phase runner resumed from 78 accepted slots.

At 17:17 UTC, v2a had stopped at 85 accepted slots. The rejected CPU attempt
`612c8547-754b-43cc-a0c5-97f5a00d420c` confirmed contention: median quota fraction
0.9842, 58 valid samples, coverage 0.9667. Its healthy baseline checkout p95 was
2597.72 ms, while injection p95 was 886.30 ms with no non-201 responses. It failed
the unchanged relative checkout-effect gate, which required 3896.58 ms here.
This diagnostic reads only the rejected observation and its baseline. It does
not fit thresholds, inspect accepted validation data, or select a policy.
The attempt remains rejected and outside the accepted dataset. Its diagnosis,
failure reports, schedule snapshot and health-check report are saved under
`runs/adaptive_v2a/effect-rejections/612c8547-754b-43cc-a0c5-97f5a00d420c/`.

Both controllers had exited and cleanup had passed. The original payment quota
and period remained 250000/1000000, and the helper PID file was absent. Five
healthy checkouts and all running resource states passed before resuming.
The existing bounded retry wrapper now also permits this specific CPU effect
rejection after checking its rejected receipt and confirmed cleanup. It saves
the failure audit, verifies lab health and collects a fresh incident. It retains
the limit of three failures without accepted progress. Fifteen focused checks
passed, including refusal to retry CPU rejection with unconfirmed cleanup.
No helper, quota, timing, evidence or recovery threshold changed. Collection
and the phase runner resumed at 85 accepted slots; physical policy results are
still pending.

At 22:16 UTC, capture had stopped at 138 accepted slots after Docker inspection
timed out during a healthy baseline. Attempt
`a8ae150a-0378-4ec3-8d8a-663a94e517c1` remains recovery-failed and outside the
accepted dataset. No fault was injected. The phase runner stayed alive, waiting.
Docker Desktop's engine and status calls hung. Relaunch exposed inaccessible
runtime sockets for its inference and secrets-engine services. Refreshing both
runtime directories together restored the engine. Old runtime directories were
preserved as backups; no container storage, settings or secrets were changed.

The engine restart left lab containers exited. Only existing containers with the
verified `rca-lab` project label were started. Every registered container ID and
the original v2a helper hash matched the frozen inventory. All lab resources ran
and five semantic checkouts passed before resuming capture. Recovery evidence
is saved under `runs/adaptive_v2a/engine-recovery/` for the failed incident.
Both CLI entry points now save subprocess failures as terminal reports with
their error type and return a failure exit code. Ten focused CLI and capture
recovery checks passed, including the actual adaptive timeout entry point.
Collection resumed at 138 accepted slots with the existing phase runner.

At 23:24 UTC, both controllers were absent without terminal failure reports.
Collection had reached 148 accepted slots. The latest accepted attempt and its
recovery receipt were saved. The recorded capture owner was dead, and no lab
block remained. Manual recovery verified all resources running and five healthy
checkouts before clearing the stale lock. No recovery command changed the lab.
The prior phase report, schedule snapshot and health check are preserved under
`runs/adaptive_v2a/controller-recovery/20261007-2324/`.

The original-helper capture and the sole waiting phase runner were restarted as
hidden background processes, with standard output and error saved in dated log
files under their existing run directories. This avoids dependence on terminal
session handles and retains failure traces if the app's tool sessions disappear.
Collection resumed from 148 accepted slots. Helper pins, lab identities, accepted
captures and evidence gates remain unchanged.

V2a physical collection completed at 00:21 UTC on October 8, 05:51 IST. All 160
accepted captures are saved: 80 training, 40 validation and 40 reserved test.
Completion verification checked every public measurement and receipt checksum,
frozen schedule assignment, unique physical episode, accepted effect and confirmed
recovery. It opened no private labels. Its report is
`runs/adaptive_v2a/capture-completion-verification.json`. No capture lock or lab
block remains. Collector revisions and recovery interruptions documented above
remain limits of this cohort.

The existing runner advanced through the frozen split, source snapshot,
training-only encoder and model fit, and validation full-evidence reference.
The reference correctly classified all 40 validation incidents. This is a
diagnostic check using all evidence, not a limited-budget algorithm result.
The package runtime is now frozen at
`54322988ac5c35261153217f092d07d4a261dfe36d76098870652f824ad9ef46`, with source
snapshot hash `8e7d72a72495350dbfc97f3ac1d47d652623276c3a3c243038ffb388751fa6c2`.
Expected SARSA seed 11 completed 10000 replay-training episodes; seed 22 was
active at the completion check. Remaining seeds, the second learner, frozen
comparisons, separate sweeps and later physical profiles continue in the same
runner. No algorithm promotion or reserved-test result is claimed yet.

The v2a main comparison is now complete. Both learners finished all five seeds
with 10000 replay-training episodes each. Saved validation selection and reserved
test reports passed, and their frozen selection hashes match. The runtime hash
still matches the immutable source snapshot. The reserved test contains 40
incidents in ten independent sessions. Verification and compact metrics are
saved in `runs/adaptive_v2a/main-comparison-verification.json`.

The selected fixed order, Expected SARSA and two-step SARSA all reached 100%
test accuracy, return 0.8625, 2.25 probes and 2.75 credits per incident. Every
learner seed matched these values. Pure policy choice time averaged 0.082 ms
for either learner, versus 0.00284 ms for the selected fixed order. Hybrids used
no fallback on this cohort. VOI and DP reached 100% accuracy with 1.5 probes,
but spent three credits and returned 0.85. Their choice times were 8.26 ms and
337.96 ms. The transferred dependency rule reached 75% accuracy and return
0.6625; the generalized rule reached 75% and return 0.5875.

Neither learner met the promotion target over the selected fixed baseline.
Their paired return advantage and session bootstrap interval were both zero.
The challenge report passed execution checks, but its outages are synthetic and
add no physical incidents. Evidence and cost sweeps are still running, so the
whole v2a profile and later physical profiles remain unfinished. These results
do not establish performance for the later latency and memory profiles.

The seven v2a evidence sweeps completed by the 05:36 UTC check. Verification
checked their ten training seeds each, frozen runtime and selection identities,
saved reserved comparisons and synthetic challenge reports. The report is
`runs/adaptive_v2a/evidence-sweep-verification.json`. All variants used the same
raw cohort and retained all four causes, with separate models and checkpoints.

Status alone reached 75% reserved accuracy. Adding CPU to status reached 100%,
return 0.8625 and 2.25 probes. Adding request errors and logs retained those
values. Without status, CPU with logs and request errors still reached 100%,
return 0.85 and 1.5 probes. Logs with request errors reached 75% accuracy; logs
alone reached 50%. The diagnostic gates for status alone, logs with request
errors, and logs alone failed and remain ineligible for promotion. Successful
report execution does not mean these evidence gates passed.

No learner passed promotion over its selected fixed baseline in any evidence
variant. Cost sweeps are still active, and the full v2a profile is unfinished.
These sweeps compare evidence on the same physical incidents; they add no new
physical captures and establish no latency or memory profile results.

The user requested v2b immediately on October 8. The waiting phase controller
was replaced with a single controller starting at v2b; the existing v2a offline
cost-sweep process continues to finish its declared work. The new controller
retains that pending profile and checks its completion before declaring the
whole sequence complete. No second lab controller or capture was started.
The phase-controller change affects scheduling only; the package runtime and
v2a model/checkpoint identities remain frozen.

V2b includes checkout latency evidence and the explicit cause
`edge/orders/payment`, retaining status, CPU, request errors and application
logs. Its declared cohort contains 200 incidents with independent session roles
and separate dataset, encoder, model and checkpoint paths. Preparation installed
the pinned lease helper images between cohorts and retained the earlier lab
reports. The post-install startup check recovered and confirmed five semantic
checkouts. The controller now repeats checkout, cart, inventory and physical
CPU, latency and memory engineering checks before starting v2b capture. Logs
are saved as `runs/adaptive_v2/phase-controller-v2b-20261008.stdout.log` and
the matching stderr log. Preparation is active; capture and policy results are
not yet claimed.
The checkout, cart and inventory checks completed, and the CPU engineering
check is active. Twenty-seven focused metric-contract and leased-helper tests
passed. The revised phase controller also passed Python syntax compilation.

V2b physical readiness completed on October 8 at 06:23 UTC. Checkout, cart,
inventory, leased CPU, payment-path latency and memory engineering checks all
passed. The profile inventory confirms all ten required actions, including
checkout latency. The 800 ms payment-path pilot measured a median checkout
latency of 921.85 ms and confirmed recovery before collection began.

V2b capture is now active under the single lab lock owned by PID 25612. The
first of 200 declared incidents is saved. Public measurements and the accepted
receipt passed checksum verification, and the receipt confirms recovery.
Private labels were not opened. The verification record is
`runs/adaptive_v2b/readiness-verification.json`. This completes readiness only;
collection, training and policy comparisons remain pending. The existing v2a
offline cost sweeps continue under their frozen identities.

At the 06:34 UTC follow-up, v2b had stopped after its second attempted incident.
The orders container restarted, but checkout recovery exceeded the declared
window. That attempt is RECOVERY_FAILED and remains outside accepted captures.
Its receipt, block, schedule and controller failure reports are preserved in
`runs/adaptive_v2b/recovery-retries/45658798-9215-4817-97a7-84c83cd0d327`.

Manual recovery verified the original orders container identity, five successful
semantic checkouts and all registered containers running. It reset the workload
and removed the recovery block only after those checks passed. The sole phase
controller resumed at v2b, and capture PID 16184 owns the lab lock. The first
accepted incident remains intact. Schedule, helper pins, container identities,
acceptance criteria and frozen runtime remain unchanged. V2a offline cost
sweeps continue. New controller logs use the suffix `20261008-0636`.

At the 07:39 UTC follow-up, five v2b incidents were accepted. Another orders
restart exceeded the checkout recovery window. The rejected attempt and its
failure reports are preserved under `recovery-retries/58b839c7-baa6-4154-989b-eda8b77f868d`.
Manual recovery again verified the original container identity, five successful
checkouts and all registered containers running before clearing the block.
All five accepted public measurement and receipt checksums passed verification.
No private labels were opened.

The scheduling scripts now apply the existing bounded recovery wrapper to v2b
and v2c using their own report and archive paths. It retries known checkout
recovery failures only after confirmed cleanup, and verified CPU effect
rejections only after confirmed recovery. It stops after three failed attempts
without accepted progress. The rejected attempt is never adopted into the
accepted schedule. No acceptance window or evidence gate was loosened.
Eleven publication and recovery tests passed, including both leased profiles
and refusal to retry after unconfirmed recovery. Python syntax checks passed.
The frozen package runtime still hashes to
`54322988ac5c35261153217f092d07d4a261dfe36d76098870652f824ad9ef46`.

The single phase controller resumed with logs ending `20261008-0742`.
The capture subprocess writes `runs/adaptive_v2b/capture-resume-attempt.json`;
`capture-evaluation.json` becomes the final success or terminal failure report
when the wrapper exits. An old failure report therefore does not establish a
new failure while that wrapper and its capture are alive. V2a offline cost
sweeps progressed to the third declared cost variant and remain active.

At the 08:35 UTC follow-up, v2b was stopped at five accepted incidents. Docker's
orders-stop command timed out, and the bounded recovery check failed. Attempt
`e8ef1eb8-9b92-4f10-bcb1-86df35f19144` remains outside accepted data.
Docker Desktop's engine calls hung. Relaunch exposed inaccessible zero-byte
runtime sockets again. Both runtime directories were preserved with suffix
`stalled-20261008-0840` and refreshed together. Container storage and settings
were retained. The user subsequently confirmed Docker Desktop was running.

Docker 29.6.2 responded. All seven registered container IDs matched the frozen
inventory, and only existing containers belonging to `rca-lab` were started.
The first recovery check encountered a compose startup timeout. The next check
passed five semantic checkouts and confirmed all registered containers running,
reset the workload and cleared the block. Identity and recovery reports are
saved under `runs/adaptive_v2b/engine-recovery/e8ef1eb8-9b92-4f10-bcb1-86df35f19144`.
The sole v2b phase controller resumed with logs ending `20261008-0845`.
Accepted data, schedules, helper pins and frozen runtime remain unchanged.
V2a offline cost sweeps remain active on the fourth declared cost variant.

At 10:34 UTC, v2b capture continued normally with 26 accepted incidents and
one active lab lock. All declared v2a sweep subprocesses had exited. Eleven
sweep profiles completed, but the third cost variant `ad9a355bea83` failed
while saving its synthetic challenge report with `No space left on device`.
Its training and saved reserved comparison completed before that failure.
The current filesystem has about 13 GB free.

A report-only recovery process is regenerating that synthetic challenge from
its existing frozen config, dataset, model and policy manifest. Original failure
and progress reports are archived in that variant's `challenge-recovery` folder.
It verifies the unchanged runtime before and after the challenge, and requires
all twelve saved sweep phase reports to pass before marking v2a complete.
It does not repeat fitting, policy training or reserved comparisons. Logs are
`runs/adaptive_v2a/cost-challenge-recovery.stdout.log` and the matching stderr
log. The recovery challenge is active; v2a completion is not yet claimed.

V2a completed by the 11:43 UTC follow-up. The report-only repair regenerated
the failed synthetic challenge successfully. All seven evidence sweeps and
five cost sweeps now have completed phase reports. Cost-sweep verification
checked all fifty training runs at 10,000 episodes each, saved reserved
comparison identities, unchanged runtime and synthetic challenge reports.
The audit is `runs/adaptive_v2a/cost-sweep-verification.json`.

For cost penalties 0, 0.025, 0.05, 0.1 and 0.2, validation selected the same
fixed policy. Its reserved accuracy was 100% and average probes were 2.25
throughout. Returns were 1.0, 0.93125, 0.8625, 0.725 and 0.45 respectively.
These scores fall as the declared probe cost rises. No RL method passed the
promotion target in any cost variant. The sweeps share the same 160 physical
incidents, with 40 reserved incidents in ten independent sessions. Synthetic
challenges add no physical incidents. These results do not establish latency
or memory profile performance.

V2b continues under the existing controller and lab lock, with forty accepted
incidents at this check. V2c remains pending. The overall phase controller keeps
its `prior_profile_running` field until the later profiles finish, but the v2a
phase report now confirms completion; no v2a process remains necessary.

At the 15:49 UTC follow-up, v2b had stopped at eighty accepted incidents after
a healthy baseline failed. Attempt `bce9dbdf-2943-46ac-a794-5a5fb4b9a3d3`
is REJECTED with recovery confirmed and no injector receipt. No fault was
injected. Its failure, schedule and attempt receipt are archived under its
`runs/adaptive_v2b/recovery-retries` directory.
An independent recovery check passed five semantic checkouts and confirmed all
registered containers running, then reset the workload. No container mutation
was needed. The single controller resumed with logs ending `20261008-1554`.
The baseline gate, schedule, helper pins and accepted data remain unchanged.

At the 16:44 UTC follow-up, v2b was still at eighty accepted incidents after
another rejected healthy baseline. Attempt `f110279e-447c-4090-9e00-99fe72d08489`
had forty-four completed checkouts and one transport timeout. Recovery was
confirmed, and no fault was injected. The failed attempt and schedule remain
archived under its `recovery-retries` directory.

The scheduling wrapper now also retries this specific baseline failure after
an independent healthy checkout check. Eligibility requires a rejected receipt,
confirmed recovery, no injector receipt and absence from the accepted schedule.
The existing three-failure bound still applies. The all-success baseline gate
is unchanged. Thirteen recovery tests passed and the frozen runtime hash is
unchanged. The first independent health check failed; diagnostic requests then
succeeded, and the repeated five-checkout health check passed without container
changes. The single controller resumed with logs ending `20261008-1654`.

At the 21:59 UTC follow-up, v2b had stopped at 130 accepted incidents after an
orders fault. Attempt `bf5d4db5-4c49-4714-8d41-f748979fbd22` remains recovery-failed
and outside accepted data. The automatic manual-recovery subprocess timed out
while resolving Docker compose state. Its failed capture report, receipt,
schedule and inventory were preserved under the incident's `engine-recovery`
directory. Docker 29.6.2 responded at this follow-up; no engine restart was
needed. The owned recovery script verified the original orders identity,
started that existing container and passed five semantic checkouts with all
registered resources running. It reset the workload and cleared the block.
The single controller resumed with logs ending `20261008-2202`. Accepted data,
helper pins, schedule and gates remain unchanged.

At the 22:55 UTC follow-up, v2b had stopped at 131 accepted incidents.
Payment CPU helper cleanup timed out after fifteen seconds. Attempt
`aa0c05f6-091e-4335-a259-8624e1f090ae` remains recovery-failed outside accepted
data. Its failure, receipt, block and schedule are preserved under its
`recovery-retries` directory. Owned recovery verified the original payment
container and recorded helper lease, ran lease-specific cleanup successfully,
and passed five semantic checkouts with all registered containers running.
It reset the workload and cleared the block before the single phase controller
resumed with logs ending `20261008-2257`. Helper pins, container identities,
accepted data, schedule and acceptance gates remain unchanged.

V2b collection completed before the October 9 05:05 UTC follow-up. All two
hundred accepted public captures and receipt checksums passed verification,
with unique physical episodes, measured acceptance and confirmed recovery.
Session roles retain one hundred training, fifty validation and fifty reserved
test incidents. No private raw labels were opened during this audit. The frozen
runtime is unchanged. Both learners completed five seeds at ten thousand
training episodes each. The saved validation and reserved comparison identities
match the frozen manifest. The audit is
`runs/adaptive_v2b/capture-and-main-comparison-verification.json`.

Full-evidence validation accuracy was 98% and its diagnostic gate passed.
Validation selected DP. On fifty reserved incidents in ten independent sessions,
DP and all five seeds of both learners, in tabular and hybrid form, reached
100% accuracy, return 0.84 and 1.8 average probes at 3.2 credits. The selected
fixed baseline `fixed-13-0.8` reached 100% accuracy, return 0.83 and 2.2 probes
at 3.4 credits. Learners gained 0.01 return over that baseline, below the declared
0.02 promotion requirement. No learner passed promotion. VOI reached 100%
accuracy, return 0.8 and 2.0 probes. The dependency rule reached 60% accuracy.
The main synthetic challenge is active, followed by evidence and cost sweeps.
These completed collection and comparison stages do not mark the whole v2b
profile complete. V2c remains pending.

The v2b main synthetic challenge completed by the October 9 06:07 UTC check.
Verification confirmed six scenarios, all frozen policy definitions and fifty
source snapshots per method in each scenario. No additional physical incidents
were collected. The audit is `runs/adaptive_v2b/main-challenge-verification.json`.
For the validation-selected DP policy, logs outage retained 100% accuracy;
request-metric outage and stats outage each reached 80%; loss of all telemetry
reached 60%. Double and half probe prices retained 100% accuracy, with returns
0.68 and 0.91. These are synthetic replay checks, not physical outage results.
The evidence sweep is now training under the existing controller. The full
v2b profile, its cost sweep and the later v2c profile remain unfinished.

At the October 9 14:18 UTC check, the user-reported disk exhaustion had cleared
and C: had about 25 GB free. V2b offline cost training remained active. Five
evidence variants had completed, while disk exhaustion interrupted later
variants during fitting or training, and one lacked a saved phase report.
These failed reports are retained; no whole evidence-sweep completion is claimed.

Docker still reported unable to start. With no physical capture active, its
verified zero-byte runtime socket directories were preserved with suffix
`stalled-20261009-1420`, and Docker Desktop was relaunched. Docker 29.6.2 then
responded. Container storage and settings were retained. Lab identity and
checkout checks must pass before v2c collection resumes.

A recovery process now waits for the original v2b offline runner PID 8572 to
exit. It then reruns only failed or missing declared variants, resumes saved
training checkpoints, retains existing fitted models and frozen comparisons,
and verifies all variant phase reports and the unchanged runtime before marking
v2b complete. It never starts a physical capture. Logs are
`runs/adaptive_v2b/disk-sweep-recovery.stdout.log` and the matching stderr log.
V2c remains pending until sweep recovery and lab health verification finish.

At the October 9 15:21 UTC follow-up, the first v2b cost variant had reached its synthetic challenge. The recovery waiter had exited when its PowerShell process check timed out. The waiter now holds a native Windows handle to the original sweep process and waits for that process to exit before starting any repair. Its replacement launcher is 45240, with logs ending 20261009-205222. Verification found both waiter processes alive, empty recovery stderr and the cost challenge active. Accepted captures, fitted models and frozen runtime identities were unchanged. Full v2b and v2c remain pending.

User requirement added October 9: after the four disk-interrupted evidence variants recover, and before any v2c collection starts, deliver a detailed but concise report at undergraduate level covering all completed work. Include algorithm benchmarks for accuracy, return/score, computation per action choice and average probes, seed variation, evidence and cost sweeps, synthetic challenge results, and promotion limits. Use verified reports and distinguish physical captures from replay and training. Finish and verify the v2b cost sweep as well so the report covers completed v2b. Record report delivery before starting v2c; no additional user approval was requested.

Project paused at the user's request on October 10 IST. Automatic follow-up is PAUSED. The recovery waiter was stopped first, then all owned phase, experiment and challenge processes were stopped and their absence verified. All accepted captures, fitted models, saved training checkpoints and completed reports remain in place. Two v2b cost variants are complete; the third challenge was interrupted. Its frozen comparisons and training are retained. Four disk-interrupted evidence variants still need recovery. The process and progress snapshot is under runs/adaptive_v2/pause-20261010-013617, and saved artifact checksums and restart instructions are in runs/adaptive_v2/pause-checkpoint.json. Do not resume without explicit user instruction. The requested benchmark report must still be delivered before v2c starts.

The user explicitly resumed the project on October 10 IST. All 569 saved artifact checksums and the frozen runtime hash matched the pause manifest. No prior experiment controller remained, and about 20 GB was available. One offline recovery runner started with launcher 48004 and logs runs/adaptive_v2b/resume-20261010-135634. It skips complete variants, resumes saved checkpoints and retains fitted models and frozen comparisons. The first unfinished evidence variant entered its full-evidence reference check. Automatic follow-up is ACTIVE again. V2c remains gated on complete v2b verification and delivery of the requested benchmark report.

At the October 10 14:58 IST follow-up, two interrupted evidence variants had recovered, bringing evidence completion to seven of nine. The next variant failed encoding because a partial pending bundle from disk exhaustion remained. Its dataset had no sealed splits, fitted model or training checkpoints. The entire unsealed derived directory was preserved under runs/adaptive_v2b/disk-sweep-recovery/partial-derived-dataset-8269cc00a53a before rebuilding from unchanged raw captures. Recovery archives now use unique attempt directories to preserve earlier failure reports. A single replacement runner started with launcher 48996 and logs ending 20261010-145953. Two cost variants remain complete; their results were retained. Frozen package identity, accepted captures and existing models/checkpoints remain unchanged. The pre-v2c report is still pending.

V2b evidence sweep completed and passed verification at the October 10 15:54 IST follow-up. All nine variants retain the same raw cohort and cause scope; 90 training runs completed, and frozen comparison identities plus six synthetic challenge definitions per variant matched. Audit: runs/adaptive_v2b/evidence-sweep-verification.json. Selected reserved accuracy was 60% with status alone, 80% with CPU and status, 100% when latency and CPU were both available, 80% with latency/logs/request errors, and 60% with latency/logs or latency alone. Seven subsets failed the diagnostic gate and retain that result. In the declared CPU+latency+logs+request_errors subset, all five seeds of both learners in tabular and hybrid form passed the saved promotion requirement: accuracy 100%, return 0.82, 1.8 probes versus selected fixed-11-0.8 at return 0.80 and 2.0 probes. Paired return advantage was 0.02 over ten independent sessions with a positive saved interval. This applies to that declared subset only; full-evidence v2b still has no learner promotion. Two cost variants are complete and the third synthetic challenge remains active. The full pre-v2c benchmark report will follow completed cost verification.

V2b completed by the October 10 17:58 IST follow-up. All five cost variants passed verification of frozen comparisons, six synthetic challenge definitions and 50 complete training runs, with unchanged runtime and shared raw cohort. Audit: runs/adaptive_v2b/cost-sweep-verification.json. At cost weight 0.2, all Expected SARSA seeds and three two-step seeds passed promotion against the selected fixed baseline; two-step seeds 11 and 22 deferred and did not pass. Both full profiles v2a and v2b are complete, with 360 accepted physical incidents and 280 declared learner training runs. The requested detailed undergraduate-level report is delivered at docs/pre_v2c_benchmark_report_2026-10-10.md. Benchmark extraction is saved in runs/adaptive_v2/pre-v2c-benchmark-data.json. V2c has not started. Its next step is identity, cleanup and health verification before physical collection. The offline recovery controller has exited normally.

V2c started on October 10 at 18:58 IST after report delivery. No prior experiment controller was active. Preflight matched registered container identities, payment/carts helper image pins, proxy identity/image and helper binary hash. Eighteen existing lab containers were started without replacement. Five consecutive semantic checkouts and twenty cart operations passed, with the proxy reporting zero delay. Receipts are under runs/adaptive_v2c/preflight-identities.json, preflight-health.json and preflight-cart.json. One phase controller launched with PID 5264 and logs ending phase-controller-v2c-20261010-185831. It runs the unchanged declared 240-incident v2c cohort, then fitting, five-seed learner training, frozen comparisons, challenges and separate sweeps. Runner scheduling now retains completed v2a/v2b and refuses a v2c start until their completion and report-delivery gate are recorded. Frozen package identity, helpers and accepted earlier captures remain unchanged.

V2c stopped at 35 accepted incidents during the October 10 evening check. Memory attempt 07b8075d-26ec-4f09-8a28-a3208f96e30d was rejected with retained memory pressure absent. Its saved samples were about 448.7 MB against a 536.9 MB limit, but only 34 samples were recorded, leaving the required coverage unconfirmed. Cleanup was confirmed and no lab block remained. The failed receipt, report and schedule were preserved under its recovery-retries directory. The recovery wrapper now permits bounded retries of this exact rejection only for v2c, an owned carts memory helper lease, a rejected unaccepted incident and confirmed cleanup. Seventeen focused tests passed. Five healthy checkouts and all running resource states were verified before one controller resumed. Package source, helper pins, injector settings, evidence thresholds and accepted captures were unchanged.

At the October 11 04:08 IST follow-up, v2c had stopped at 97 accepted captures. Uninjected healthy control 59254c09-8c56-47ac-b12a-9ae7f38c2f33 was rejected with healthy control degraded and confirmed cleanup. Its failure, receipt and schedule were archived under recovery-retries. Bounded retries now also recognize this exact error only for rejected unaccepted captures with confirmed cleanup and no injector, injection or allocation control. Nineteen focused recovery tests passed. Five healthy semantic checkouts and running registered states passed before one controller resumed. No accepted data, source identity, helper pins or evidence gates changed.

At the October 11 13:19 IST follow-up, v2c stopped at 203 accepted captures when Docker timed out during payment CPU helper cleanup for incident 3d49ffec-009e-4f1b-8039-a052ac26899f. The interrupted receipt, failure, lab marker and schedule were preserved under its recovery-retries directory. After confirming the capture owner and phase controller had exited, lease-specific helper cleanup passed, followed by five healthy semantic checkouts and running registered resource states. Source identity, helper pins and all existing container identities matched; no containers needed starting. One replacement phase controller resumed with launcher 44840 and logs ending 20261011-131945. Accepted captures and evidence gates were retained.

V2c physical collection completed by October 11 17:19 IST. The audit verified 240 unique accepted physical episodes, all public bundle checksums, confirmed recovery, 120 training, 60 validation and 60 test incidents, unchanged runtime and no remaining lab lock/block. Audit: runs/adaptive_v2c/capture-completion-verification.json. Fitting and encoding completed, but full-evidence validation reached 40 of 60 correct, 66.7%, and the declared gate correctly stopped policy training. Healthy and edge-delay cases were each misclassified as carts pressure in all ten validation sessions. Investigation using training and validation only found the fitted carts pressure threshold was 57.734% of the limit; ordinary validation memory was 57.75% to 58.24%, while injected pressure remained about 83.5%. This small normal drift changed every validation memory category to PRESSURE. Diagnostic audit: runs/adaptive_v2c/diagnostic-gate-investigation.json. Reserved test responses remain unopened; no training, comparison or promotion is claimed for v2c. Preserve this failed frozen experiment and its encoder/model identities. Continue investigating a separately declared, training-only measurement revision before opening any reserved comparison; do not bypass or lower the diagnostic gate.

The requested v2c findings report was delivered on October 11 at docs/v2c_findings_and_benchmark_report_2026-10-11.md. It distinguishes completed physical collection from the unfinished experiment, explains the 66.7% diagnostic gate failure and memory-category drift, and repeats the verified v2b algorithm benchmarks as context without presenting them as v2c results. V2c policy timing, score, probes, learner seeds, sweeps and promotion remain unmeasured.

On October 11 at 17:35 IST, the user-authorized memory revision reused all saved v2c captures in separate paths. scripts/revise_v2c_memory.py records its fitting rule before reading training measurements: midpoint between the maximum healthy and minimum verified carts-pressure training-window median, fail closed on overlap, with at least five independent sessions per group. Twenty healthy and twenty fault training windows from twenty sessions per group yielded a 70.5957% threshold, between 57.7344% and 83.4570%. Other encoder values, runtime source, physical captures and original failed artifacts were retained. Four focused checks passed. The declared revision and fitter hash are in runs/adaptive_v2c_memory_separation_r1/revision-declaration.json; fit provenance is in memory-boundary-fit.json. A startup task-name registration error was archived before correcting the config to retain the registered physical task; no measurements or fitting rule changed. Revised encoding and model fitting completed. Full-evidence validation passed with 60 of 60 correct, ten of ten for each class. The unchanged 90% gate passed, and one offline controller, actual PID 37740, started five-seed training with logs ending controller-20261011-173516. Reserved-test policy evaluation and sweeps remain pending. The original 66.7% failure remains recorded; validation success for this development revision is not a reserved-test result. No physical recollection was needed.

Revised v2c main training completed: both Expected SARSA and two-step SARSA finished seeds 11, 22, 33, 44 and 55, each at 10,000 replay episodes. All ten saved reports passed and recorded completion. The same offline controller is now comparing policies on validation before freezing policy definitions and evaluating the reserved cohort. No new physical incidents were collected.

The revised v2c main reserved comparison completed and passed audit on October 11. Validation selected VOI; the frozen manifest hash and unchanged runtime matched the sixty-incident test report and ten completed training runs. VOI and DP each reached 93.33% accuracy, return 0.72 and 2.3 average probes. Five-seed mean results were Expected SARSA tabular 76.67%, return 0.5008, 3.9 probes; hybrid 81.67%, 0.5562, 3.6467 probes; two-step SARSA tabular 79.67%, 0.5293, 3.7167 probes; hybrid 88.33%, 0.6088, 3.56 probes. No learner passed performance promotion. Audit: runs/adaptive_v2c_memory_separation_r1/main-comparison-verification.json. The main synthetic challenge is running, followed by declared evidence and cost sweeps. This main comparison does not mark the full revision complete.

The user-requested method and training explanation was delivered at docs/algorithm_working_and_training_report_2026-10-11.md. It explains Expected SARSA, two-step SARSA, depth-1 VOI, depth-2 bounded DP and the visit-count hybrid wrapper, linking theory to the implemented update/search rules. The report includes verified main-test decision timings, accuracy, return, probes and credits, and documents replay sampling, train/validation/test roles, exploration, checkpoint selection and frozen evaluation. Timing extraction is saved in runs/adaptive_v2c_memory_separation_r1/method-report-metrics.json. Challenges and sweeps remain separate pending work.

The revised v2c main synthetic challenge completed and passed verification on October 11. All six scenarios used the frozen definitions and sixty source snapshots per method, with no additional physical incidents. For validation-selected VOI, logs outage retained 93.33% accuracy and return 0.72; request-metrics outage reached 83.33% and 0.525; stats outage reached 60% and 0.4167; all-telemetry outage reached 33.33% and 0.175. Double prices reached 85% accuracy and 0.4833 return; half prices reached 93.33% and 0.8167. These are synthetic replay outcomes, not physical outage captures. Audit: runs/adaptive_v2c_memory_separation_r1/main-challenge-verification.json. The existing controller started the declared evidence sweep; the first status subset has trained and entered validation comparison. Full evidence and cost sweep completion remain pending.

Revised v2c evidence sweep completed and passed audit at the October 13 IST check. All eleven variants, 110 complete five-seed training runs, frozen reserved comparisons and six synthetic challenges per variant matched the unchanged runtime and shared raw cohort. Audit: runs/adaptive_v2c_memory_separation_r1/evidence-sweep-verification.json. Full evidence selected VOI at 93.33% accuracy, return 0.72 and 2.3 probes. Removing memory reduced selected accuracy to 76.67%; removing status also reached 76.67%. Status alone reached 33.33%, and memory alone reached 18.33%. Ten reduced subsets failed the diagnostic gate; their comparative results remain recorded without promotion eligibility. No learner passed promotion in any subset. These are replay and training results using existing captures, with no new physical incidents. The existing controller started the first of five cost variants and is running its synthetic challenges; cost sweep completion remains pending.

Project paused at the user's request on October 13 IST. All owned experiment controllers and children were stopped and their absence verified. Saved artifacts remain intact; 533 model, checkpoint and report checksums are recorded under runs/adaptive_v2/pause-20261013-112039. Main revised v2c and the eleven evidence variants are complete. Three cost variants are complete; the fourth retains completed training and frozen comparisons, but its synthetic challenge was interrupted. The fifth remains pending. Resume instructions are saved beside the checksum manifest. Resume only on explicit user instruction.

The user explicitly resumed the project on October 13 IST. All 533 saved artifact checksums matched, no prior controller remained, and about 12.4 GB was free. One offline recovery runner started with actual PID 26304 and logs ending resume-20261013-183624. It skips completed variants and preserves fitted models, frozen comparisons and checkpoints. The fourth cost variant resumed its interrupted synthetic challenge; the fifth remains pending. Automatic follow-up is ACTIVE again. No new physical collection is required.

Revised v2c completed on October 14 IST. The cost audit verified all five variants, fifty complete training runs, frozen reserved comparisons and six synthetic challenges per variant with unchanged runtime. At cost weights 0, 0.025 and 0.05 selected accuracy was 93.33%; at 0.1 it was 85%; at 0.2 VOI deferred on every case, with zero diagnosis accuracy, zero probes and return 0.25. No revised v2c learner passed promotion. Main, evidence and cost runs now total 170 training runs and 1.7 million replay episodes; all three profiles total 600 accepted physical captures and 450 cohort training runs. Saved main comparisons also contain 190 frozen-method mixture sweeps. The final simple benchmark report is docs/revised_v2c_benchmark_report_2026-10-14.md. All experiment controllers exited; automatic follow-up is PAUSED because authorized implementation and physical runs are complete. The original 66.7% diagnostic failure and its frozen identities remain preserved. Completion records execution, not successful performance promotion.
