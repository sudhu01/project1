# Remaining faults, multi-cause diagnosis and DQN implementation plan

14 October 2026. Repository: `team-graph-rca`. This document proposes work; it does not start collection, training or background runners.

The user corrected the requested scope from three faults to all five remaining upstream scenarios. Keep Expected SARSA, two-step SARSA, VOI and bounded DP. First make diagnosis and scoring support multiple causes, then add vanilla DQN as a separately declared policy comparison. A neural policy chooses evidence actions; it does not replace the shared diagnostic model.

## 1. Scope and experimental counts

| New scenario | Physical intervention | Diagnosis truth | Main complication |
|---|---|---|---|
| Payment stopped, called bad deployment upstream | Stop the exact payment container | `{resource/payment}` | Same root as CPU contention, different mechanism |
| Orders database stopped | Stop the exact orders-db container | `{resource/orders-db}` | Order failure can resemble orders service failure |
| User/auth service stopped | Stop the exact user container | `{resource/user}` | An existing authenticated session can hide new login failure |
| Shipping stopped | Stop the exact shipping container | `{resource/shipping}` | Checkout success does not prove fulfilment success |
| Simultaneous database failure and payment CPU contention | Stop catalogue-db and start the bounded payment CPU lease during one common window | `{resource/catalogue-db, resource/payment}` | One fault can mask the other fault's customer symptoms |

Preserve the five existing fault scenarios and healthy operation. The expanded physical schedule has **eleven scenario strata**. It has **ten distinct diagnosis sets**: eight singleton fault roots, one known two-root combination and the empty healthy set. Payment stop and payment CPU contention are separate strata but share one diagnosis set.

The eight atomic causes are catalogue-db, orders, payment, carts, orders-db, user, shipping and the orders-to-payment edge. The edge is a cause identity, not another service. A hypothesis is a possible set of those causes.

Primary scope supports zero, one or two causes, with only the declared catalogue-db/payment pair initially supported. It does not claim diagnosis of every possible pair, triples, arbitrary deployments or an actual application memory leak. Payment stop does not establish a defective deployed version. Shipping stop does not establish lost fulfilments until that effect is measured.

Recommended primary collection is a fresh, coherent cohort of 440 accepted captures, forty per scenario: twenty train, ten validation and ten reserved test. This retains the previous allocation and provides new evidence for every old and new condition under the expanded workload. Pilots remain separate. Existing accepted captures stay intact; their reserved labels must not become new training data.

## 2. Current code and changes required

The completed source uses `src/rca_investigation`, not the old PPO simulator. Existing v2c has six diagnosis classes, 240 accepted captures and frozen source/model/checkpoint identities. Its original encoder failed at 66.7%; the separate memory revision passed validation and completed its comparisons. Preserve both records.

Concrete restrictions found in the implementation:

- `contracts_v2.py` registers one `cause_id` and fixes a mechanism per profile cause. Payment stop is rejected by its current v2 profile even though payment CPU is supported.
- `experiment.py:label_name` returns one scalar cause. `belief.py` stores scalar class labels and a posterior over those classes.
- `environment.py` chooses one posterior class on STOP. `score_decision` checks one string against one truth string.
- `policies/dp.py` values STOP using the maximum single-class posterior. Its fallback and tie-breaking are shared policy behavior.
- `raw_schedule.py`, `capture_v2.py` and `faults_v2.py` schedule and verify the currently approved interventions. The CPU helper only targets payment; cleanup uses a controller-owned lease.
- `workload_v2.py` supports checkout and cart traffic. `ToolSpecV2` currently permits checkout as the request/latency operation, so auth and fulfilment require deliberate contracts, not invented response codes.
- `train.py` and `policies/n_step_sarsa.py` implement tabular training and deterministic checkpoint resume. There is no DQN policy or neural training dependency in `pyproject.toml`.
- `comparison.py` loads frozen definitions and validates source, dataset and checkpoint hashes. `cli.py` only exposes replay for incident investigation and does not offer general live agent execution.

Do not change old serialized v1/v2 records to mean cause sets. Create v3 incident, decision and objective records, while reusing unchanged v2 tool packets where their measurement semantics still apply. Dispatch by explicit version and test the mixed record boundary.

## 3. Proposed files and artifact layout

Names below are proposed additions. Existing modules receive explicit v3 branches or call new helpers; their v1/v2 behavior must remain stable.

| Path | Responsibility |
|---|---|
| `src/rca_investigation/contracts_v3.py` | Cause sets, mechanism records, public incident and decision records |
| `src/rca_investigation/diagnosis_sets.py` | Canonical set mapping, exact scoring, marginals and terminal utility |
| `src/rca_investigation/faults_v3.py` | New stop targets and composed injection/recovery transaction |
| `src/rca_investigation/workload_v3.py` | Fresh auth attempts, checkout and verified fulfilment operation |
| `src/rca_investigation/capture_v3.py` | Common window, independent component verification and raw receipts |
| `src/rca_investigation/state_vector.py` | Versioned numeric public state for DQN |
| `src/rca_investigation/policies/dqn.py` | Inference network and masked action choice |
| `src/rca_investigation/train_dqn.py` | Training transitions, replay buffer, target network and checkpointing |
| `src/rca_investigation/comparison.py`, `challenges.py` | New definitions, set metrics, fair comparisons and challenges |
| `config.py`, `catalogue.py`, `dataset.py`, `capture_store.py`, `encoding.py`, `raw_dataset.py`, `experiment.py` | Profile registration, version dispatch and immutable fitting |
| `cli.py`, `adaptive_cli.py`, `explain.py` | Explicit v3 paths, DQN commands and multiple-cause explanations |
| `configs/investigation_v3_single.yaml` | Ten single/healthy scenario strata and nine diagnosis sets |
| `configs/investigation_v3_multi.yaml` | Eleven strata and ten diagnosis sets, including the pair |
| `configs/resources_sockshop_v3.yaml` | Exact service/container registry and verified capabilities |
| `configs/capture_v3_pilot.yaml`, `capture_v3_evaluation.yaml` | Separate mechanics and evaluation schedules |
| `experiments/v3/experiment-declaration.json` | Scenario, utility, budget, gates and comparison settings |
| `experiments/v3/neural/pyproject.toml`, `uv.lock` | Separate pinned neural environment |
| `scripts/run_v3_phases.py` | One resumable controller that skips complete stages |
| `scripts/audit_v3_completion.py` | Capture, split, checkpoint and frozen-report audit |
| `datasets/raw_v3`, `datasets/adaptive_v3_single`, `datasets/adaptive_v3_multi` | Raw cohort and separately declared derivations |
| `runs/adaptive_v3_single`, `runs/adaptive_v3_multi`, `runs/adaptive_v3_dqn_r1` | Separate runs and method artifacts |

Use method manifests to share the same multi-cause environment, encoder and diagnostic model between SARSA and DQN. A separate DQN run directory must not silently create a different reward or dataset. Freeze common package source after all required modules are implemented, before physical evaluation or training.

## Phase 0. Freeze the experiment definition

1. Record Git revision, actual runtime source hash, dependency lock hashes, Docker image IDs, helper binary hash, resource registry and the complete old artifact index.
2. Preserve runnable old source snapshots and their dependency locks. New source will have a new runtime hash; do not weaken old frozen-manifest checks to make it look unchanged.
3. Declare the eleven scenario strata, ten diagnosis sets and permitted mechanism mixtures. Declare the pair explicitly as two roots, not a new scalar diagnosis string interpreted as one root.
4. Declare the primary episode budget of ten credits and five probes, lambda 0.05, correct utility 1, incorrect utility 0 and DEFER utility 0.25. Larger-budget experiments get separate identities.
5. Declare five learner seeds, the initial 10,000-episode training budget, validation cadence, fixed-baseline search budget, DP limits, test metrics and promotion rules.
6. Keep entry context as the same manual frontend event for every incident. Do not expose the targeted service, mechanism, number of active faults or schedule slot as an alert hint.
7. Create a status file with completed phases, hashes, process owner, log paths, accepted slots and pause state. Do not restart the completed v2 automation merely to execute this plan.

Exit check: a canonical declaration can be hashed and validated without touching Docker. No selection rule or gate is left to be invented after reserved-test results arrive.

## Phase 1. Add real cause-set contracts and scoring

Define a canonical `cause_ids` tuple sorted by registered cause identity. Empty means healthy. `DEFER` is a decision type, not a fault in the tuple. Require uniqueness, registered members and a permitted set cardinality. Validate the exact combination against the profile hypothesis registry.

Private incident records contain a mechanism per active cause and a receipt per injected component. A payment CPU case and a payment stop case therefore have the same cause set but different private mechanisms. Reject duplicate resources, contradictory mechanisms such as stopping and burning the same payment container, and mismatched component receipts.

Public incident records retain window, task, topology, encoder, catalogue, objective and entry metadata. They contain no injected cause set, mechanism count, accepted-condition name or verification thresholds. Decisions contain `decision_type`, sorted predicted causes, set confidence, evidence IDs, policy/model hashes and stop reason.

Use exact set matching for the primary terminal reward:

```text
correct singleton or pair: 1
correct healthy empty set: 1
wrong set, including a missing or extra root: 0
DEFER: 0.25
episode return = terminal utility - lambda * purchased credits
```

Report root precision, recall, F1 and Jaccard separately. For truth `{catalogue-db, payment}` and prediction `{catalogue-db}`, exact match is false, precision is 1, recall is 0.5 and Jaccard is 0.5. These secondary metrics must not quietly become training rewards. Define healthy/empty and DEFER handling explicitly; exclude empty truth from per-root recall and report healthy specificity separately.

Tests: tuple order invariance; duplicate and unknown causes rejected; healthy/DEFER distinct; correct pair, partial pair and extra root scoring; scalar v1/v2 hashes unchanged; private labels excluded from public serialization.

Exit check: both causes appear in an end-to-end synthetic result, including CLI explanation and JSON round-trip, with the exact expected return.

## Phase 2. Expand tools and workloads before fault injection

1. Verify orders-db, user and shipping in the current topology and Docker registry. Register exact container IDs and dependency edges. Keep payment status from the existing catalogue.
2. Add status tools for newly included resources. Add bounded logs only where observed data confirms a usable signal. Every additional action gets units, parameters, cost, vocabulary, source and missing-data behavior.
3. Add fresh login attempts during each observation window. Reusing a prepared authenticated session alone cannot measure a stopped auth service. Distinguish login refusal, application error, timeout and traffic-generator failure.
4. Identify a real shipping/fulfilment operation from the service API and workload behavior. Declare its success contract and measure it. If end-to-end fulfilment cannot be established, restrict the first shipping scenario to verified shipping service unavailability and state that limit.
5. Prepare accounts, baskets and checkout prerequisites during healthy baseline. Do not perform essential preparation behind a dependency after deliberately stopping it and then mislabel preparation failure as checkout evidence.
6. Add operation-specific request/latency contracts for auth and fulfilment. Missing successes become unavailable latency, not a made-up low latency. Keep failed/timeout counts in the appropriate request evidence.
7. Hold workload mix and rates fixed across all scenarios, including healthy controls. Request logs must not contain private injector names.
8. Measure evidence redundancy. A status signal that already identifies a stop may make logs unnecessary; that is a valid result, not a reason to force log purchases.

Tests: workload preparation failure distinct from application failure; zero traffic unavailable; operation labels and denominators correct; CPU packets exclude memory; unmatched log peers rejected; all declared tools return either valid evidence or a valid unavailable packet.

Exit check: healthy auth, checkout and the declared shipping operation run repeatedly, with sufficient measurement coverage. Unsupported shipping semantics block only that scenario's physical acceptance, not unrelated software work.

## Phase 3. Implement bounded single and simultaneous injection

Single stops use the exact registry ID and preserve existing container identity. Payment stop and payment CPU contention get different mechanism IDs and their own receipts. No container recreation is needed to simulate a stop.

For the two-component fault:

1. Verify baseline health and confirm no owner currently holds the lab lease.
2. Start the bounded payment CPU helper with a unique controller lease and recorded expiry. Stop the registered catalogue-db. Record both intervention times.
3. Start the observation window only after both interventions are effective and the declared settling period passes. Keep their accepted common overlap for the whole 120-second window.
4. Independently verify catalogue-db stopped throughout the window and sustained measured payment CPU contention. Verify the CPU signal even if the database failure prevents customer requests from reaching payment.
5. During engineering pilots, use declared direct target probes or traffic to establish payment behavior independently of the broken catalogue path. Apply any retained workload consistently to the evaluation cohort. These are measurement actions, not private cause hints to agents.
6. Keep component verification separate from customer symptom verification. Do not require each root to produce an individually visible frontend failure when the other root masks it.
7. On completion or any error, clean up every acquired component. Restore catalogue-db and terminate the specific helper lease. Attempt all cleanup operations even if the first one fails.
8. Confirm service states and workload health. Publish the capture only after both effects and full recovery pass. Otherwise preserve a rejected attempt or a blocked-lab record.

Recovery state records acquired leases, component actions, restoration results and ownership. Resume accepted slots only; an interrupted unaccepted attempt may be retried after verified cleanup. Never repeat accepted incidents or remove a lock merely because it is old.

Tests: second injection failure cleans up the first; partial recovery failure blocks collection; expired helper excluded from acceptance; observation overlap enforced; crash recovery uses saved leases; two active root receipts required; no other lab project/container touched.

Exit check: at least five independent mechanics pilots per new scenario, plus healthy pilots, establish the intended effects and cleanup. Pilots are not training incidents.

## Phase 4. Build synthetic multi-cause decision fixtures

Implement these fixtures before spending days collecting data:

- One status probe decisively identifies a singleton.
- Two complementary probes are needed to identify both roots.
- The first root masks the second in frontend errors, but a target probe reveals it.
- A healthy high-load control produces misleading CPU symptoms.
- One observation supports a singleton and a pair equally, making deferral rational.
- The budget is insufficient to establish both roots.
- A bought action returns unavailable and still consumes its declared credits.
- Remaining legal actions are terminal-only.

All methods use the same diagnostic model, observation mask and objective. Hand-calculate expected terminal utilities and action values for small fixtures. Verify VOI depth-one and DP depth-two behavior, then both SARSA methods and DQN.

Exit check: decision traces obey the acquisition budget and exact-set reward. DQN cannot select an invalid/repeated action, and no method gets the hidden pair label as state.

## Phase 5. Collect and seal the fresh cohort

Use twenty train sessions, ten validation sessions and ten reserved-test sessions, each containing one incident from every scenario stratum. Randomize condition order within each session using a declared seed. That produces 220 train, 110 validation and 110 test captures.

Payment has forty training incidents across its two mechanisms; each other scenario has twenty. Keep a mechanism-stratum count table as well as a diagnosis-set count table. One pair capture is one physical incident, not two independent incidents.

Baseline remains 120 seconds, settling 30 seconds, observation 120 seconds and ingestion allowance ten seconds, unless pilots justify a separately declared protocol before evaluation starts. Record failed attempts and time spent on recovery. Serialize collection through one lab lease.

Declare pause boundaries between incidents. Save an atomic accepted-slot manifest and checksums. On resume, check source, helpers, images, container identities, schedule and accepted bundles before continuing. Bound storage growth and check free space before each publication. Never prune volumes or accepted captures to make room.

Freeze session assignments before collection. Seal raw bundles, split manifests and derived artifacts. The primary fresh cohort prevents later operating-state differences from being confused with algorithm differences. A cheaper reuse study may use old training captures only under a separate declaration; it is not the primary benchmark described here.

Exit check: 440 unique accepted physical episodes, disjoint session roles, complete strata, valid public checksums, component receipts and recovery records. All later fitting reads training only.

## Phase 6. Fit and validate the shared multi-cause diagnostic model

Extend the prototype model to associate each training incident with a registered hypothesis set. Keep whole-incident response rows so the evidence correlations remain represented. Do not build pair captures by merging responses from two separate single-fault captures.

For the initial ten-hypothesis profile, retain a balanced prior across diagnosis sets. Payment's CPU and stop prototypes split its one-set prior evenly by declared mechanism mixture, rather than accidentally giving payment twice the prior of every other root. Save both the hypothesis prior and within-hypothesis mechanism weights.

Expose three interfaces:

```text
posterior_sets(history) -> probabilities over registered cause sets
predict_response(history, action) -> response probabilities
terminal_decision(history, utility) -> predicted set, confidence, expected utility
```

Optional per-root marginals are derived by summing probabilities of sets containing each root. Do not multiply independent root marginals to invent an unobserved combination. With exact-match utility, expected STOP utility remains the maximum set posterior. All policies use the same terminal decision.

Fit every new threshold and log vocabulary from training only. Carry the memory lesson forward: a healthy-only extreme boundary may sit too close to ordinary drift. Declare the separation rule before fitting, require independent healthy/fault sessions and fail on overlapping training bands. Any changed threshold/vocabulary gets a new encoder hash.

Diagnostic gate before policy training:

- At least 90% full-evidence exact-set validation accuracy overall.
- At least 90% within every scenario stratum, including the simultaneous case and all retained old conditions. Payment CPU and payment stop each have their own check.
- At least 90% healthy specificity.
- Separate payment CPU and payment stop confusion reports, plus per-root precision/recall and set calibration.

Ten validation pair incidents means one error changes pair accuracy by ten percentage points. Report the count, not just a percentage. Calibrate any posterior adjustment from training-only cross-validation; keep validation for selecting declared settings. Reserved test remains closed.

Exit check: the gate passes with immutable fitted identities. On failure, fix evidence or declare a new model revision. DQN training must not bypass the gate.

## Phase 7. Adapt retained policies to set diagnosis

Use the unchanged action design: buy evidence, STOP or DEFER. The shared decoder chooses a cause set on STOP. This isolates evidence acquisition from classifier architecture and avoids giving DQN a separate diagnosis action space.

- Expected SARSA: retain its on-policy expected next-action target and public history key; register new task/model/objective hashes and start a new Q table.
- Two-step SARSA: retain the existing two-step behavior target and terminal tail handling, with the new exact-set terminal reward.
- VOI: depth-one response search and replanning, using the new expected STOP utility.
- DP: depth-two search, same legal masks and costs, explicit 50,000-state expansion cap and reported fallback. Do not describe it as globally optimal.
- Hybrids: keep visit-count fallback separate from pure learners and label it clearly.
- Rules: retain a frozen old-rule transfer baseline and add a declared expanded rule. The expanded rule must gather enough evidence for the pair before concluding a singleton. It cannot query private labels or receive free status packets.
- Fixed sequence: keep a bounded validation search, initially 32 orders and five confidence thresholds. Its STOP uses the same multi-cause decoder.

If five probes cannot diagnose a declared pair even with the best policy, record that as a budget limitation. A larger budget belongs to a separate comparison shared by every method.

Exit check: all policies pass the multi-cause fixtures and honor the same observation, costs, terminal utility and acquisition masks. No old Q table is loaded as implicit transfer.

## Phase 8. Add vanilla DQN as a separate comparison

DQN approximates action values with a neural network and reuses training transitions in a replay buffer. Our input is a compact evidence vector, not images. Its role is to share learned acquisition behavior between similar evidence states. This is a hypothesis to test, not a promised improvement. The algorithm basis is the [original DQN paper](https://arxiv.org/abs/1312.5602) and the [official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### Public input encoding

Freeze an action order and response vocabulary. For each tool encode a not-yet-bought bit and a one-hot acquired symbol, including unavailable reasons. Add legal-action mask, remaining credits/probes, declared normalized prices, lambda, objective and shared hypothesis posterior. All existing policies already have the diagnostic posterior available, so DQN does not receive privileged information.

An unbought tool must not carry its hidden response, raw measurement or incident-wide availability outcome. Public capability masks are allowed; a collector failure learned only by buying the tool is not a free mask hint. Exclude cause truth, mechanism, injector receipt, fault count, absolute capture time, incident ID and session ID. Fix scaling constants from configuration or training only. Hash feature order, dimensions, scaling, model and catalogue.

### Initial declared training settings

| Setting | Initial proposal |
|---|---|
| Network | MLP, two hidden layers of 128 units with ReLU |
| Output | One Q value per purchased-tool action plus STOP and DEFER |
| Device | CPU, one inference thread for timing |
| Optimizer | Adam, learning rate 0.001 |
| Discount | 1.0, matching the finite episode objective |
| Loss | Huber loss |
| Replay | Uniform buffer, 50,000 transitions, batch size 64 |
| Warm-up | 1,000 training transitions before gradient updates |
| Update frequency | One gradient update per collected transition after warm-up |
| Target network | Hard copy every 1,000 gradient updates |
| Gradient norm cap | 10 |
| Exploration | Same episode schedule as SARSA, 0.2 to 0.05 over 8,000 episodes |
| Budget | 10,000 episodes per seed, five seeds |
| Validation | Every 500 episodes, no exploration or learning |

These values are proposals, not reported tuned defaults. Freeze them after fixture/pilot validation and before physical test evaluation. Record update count as well as episode count; equal episodes do not imply equal compute.

Vanilla DQN target:

```text
terminal transition: y = reward
otherwise: y = reward + gamma * max Q_target(next_state, legal_next_action)
loss = Huber(Q_online(state, chosen_action), y)
```

Mask illegal actions during both epsilon exploration and greedy selection, and before the bootstrap maximum. Branch on terminal transitions so an all-masked terminal row never produces NaN through `0 * -infinity`. The target network uses no gradients.

The current environment does not expose truth in observation; its independent training scorer supplies terminal utility. Build a transition adapter that adds the terminal reward exactly once, including automatic termination at budget/horizon. STOP creates a genuine terminal transition. Never bootstrap beyond the incident.

Use vanilla DQN in the first registered comparison. Double DQN is an optional later, separately named variant with a different update rule; do not quietly replace the maximum with online-selection/target-evaluation and still call it vanilla. See the [Double DQN paper](https://arxiv.org/abs/1509.06461) for that distinction. PPO is outside this plan.

### Dependencies and checkpoints

Use an isolated pinned neural project/environment. Do not change the original v2 lock and invalidate old frozen artifacts. Register the neural lock and Torch version in new run identities. Run all v3 inference timing comparisons in the same declared CPU environment, with warm-up, and make Torch optional for non-neural commands.

Save online weights, target weights, optimizer state, replay transitions/cursor, feature schema, sampler state, exploration counters, gradient count, Python/NumPy/Torch RNG state and all source/input hashes. Save atomically at episode boundaries. Keep `latest` for resume and validation-selected `best` for evaluation. CPU deterministic continuation is the initial reproducibility target; record any unsupported deterministic operation explicitly.

Tests: masked greedy and random choice; exact target values; no target gradients; target-sync timing; terminal reward once; replay contains training transitions only; no raw/private features; checkpoint resume reproduces a controlled uninterrupted continuation; wrong model/schema/lock rejected; inference makes no buffer/optimizer updates.

Exit check: all neural tests and fixtures pass, with finite loss/targets and reproducible CPU resume. No physical result is claimed yet.

## Phase 9. Train in stages with fixed budgets

Complete software for all methods before freezing the primary source. Then:

1. Fit a separately identified single-cause task on the ten single/healthy strata. It excludes the pair explicitly. Train both SARSA methods, five seeds each. This confirms the four new single scenarios and avoids comparing changed scope as if it were identical.
2. Fit the multi-cause task on all eleven strata. Train both SARSA methods and vanilla DQN from scratch on the same training cohort, five seeds each.
3. Evaluate hybrids separately from their parent tables. DQN has no planner fallback in its primary result.
4. Sample diagnosis sets according to the declared balanced prior, then sample payment mechanisms by the declared 50/50 mixture, then a training incident. Log actual mechanism/root frequencies.
5. Select each method/seed checkpoint by validation mean return, with the same lower-credit tie-break and fixed evaluation cadence. Report all seeds; use validation only if selecting a representative seed.
6. Log exact-set validation accuracy, pair recall, premature single-root stopping, healthy false alarms, mean return/credits/probes, action use, table coverage, hybrid fallback, DQN replay size/loss/gradient norms, training time and peak RAM.

Initial staged total is 25 training runs and 250,000 replay episodes, excluding synthetic fixtures. With at most five probes and a terminal choice, DQN has at most roughly six transitions per episode, though automatic termination often reduces this. Training replay is not new physical data.

If learning curves suggest a longer run, declare a separate validation-only extension for all learners under the same extra episode budget before opening test. Do not extend only a losing DQN after seeing its test result. A hyperparameter search needs its own bounded candidate list and compute accounting.

Exit check: complete checkpoints for all declared seeds/methods, no data leakage, unchanged identities and stable validation. Any failure resumes from the last verified checkpoint rather than restarting complete runs.

## Phase 10. Freeze and open one primary reserved comparison

Freeze the full method definitions, selected checkpoints, feature schemas, model/encoder, action costs, priors, objective, source and locks before test. Seal both single-cause and multi-cause comparison manifests before opening either reserved report, since their single-fault test incidents overlap. Use the same 110 multi-cause test incidents for every method. The single-cause task has its own 100-incident test subset and must be labelled separately. Do not call those overlapping comparisons independent replications.

Report:

- Exact cause-set accuracy overall and per scenario/mechanism.
- Pair exact accuracy, missing-second-root rate and extra-root rate.
- Per-root precision/recall/F1, healthy specificity, DEFER rate and diagnosis coverage.
- Mean return, credits and probes, with per-session distributions.
- Mean, median and p95 decision computation, forward-pass warm-up, training time, peak RAM and DP search/fallback statistics.
- Acquired-evidence failure rates and posterior calibration. A softmax or posterior confidence is not automatically calibrated.
- All five seed results, mean/spread and the declared representative selection.

Use paired session bootstrap intervals, preserving the entire multi-root incident as one observation. Do not count its two roots as two independent samples. Report the limited ten-session test support.

Primary promotion proposals to freeze in Phase 0:

1. Diagnostic gate passed.
2. At least 90% exact-set accuracy overall and on the pair, plus healthy specificity at least 90%.
3. Return advantage at least 0.02 over the validation-selected strongest nonlearning acquisition baseline, with a positive paired session interval lower bound. Keep comparison to the fixed baseline too for continuity.
4. No promotion from a single favorable seed while the family results are concealed.

A separate compute-saving claim can be evaluated with a predeclared accuracy noninferiority margin of two percentage points and lower choice computation. Report it as noninferiority/compute evidence, not as return superiority. Do not weaken the accuracy requirement when a policy defers on every incident.

Exit check: every reserved report matches the frozen identity and all methods use one scorer. Failed promotion is a valid completed result.

## Phase 11. Bound robustness work and test the multi-cause boundary

Declare a small robustness matrix first, rather than repeating dozens of costly variants without a question:

- Frozen-method source outages: logs, request metrics, container stats and all telemetry. Mark these synthetic replay perturbations.
- Frozen public probe prices: half and double. Distinguish these from retraining at a different lambda.
- Pair masking strata from independently captured common-window incidents. Separate failures to see the second root from failures to buy the relevant tool.
- Mixture reweighting between healthy, singleton and declared pair outcomes, with fixed priors and policies. This estimates an operating-mixture boundary without fitting to test.
- A declared smaller-cost and larger-budget configuration only if needed, retraining every learner and refitting task-specific baselines under separate identities.

Unseen fault combinations require a later compositional diagnosis experiment. The initial prototype model has no trained support for an arbitrary pair; DQN generalization over evidence states does not solve that limitation. Do not claim unseen-combination accuracy from artificially joining single-incident packets.

Run an evidence ablation only to answer a declared question, such as whether auth status alone suffices or whether shipping logs add value conditional on shipping status. Keep all supported roots in a subset task; record failed diagnostic gates.

Exit check: synthetic challenges, physical conditions and separately retrained variants are clearly labelled. Stop after the declared matrix and report failures.

## Phase 12. Audit, documentation and manual testing

1. Audit accepted raw and derived checksums, split/session independence, component recovery, model training IDs, fitted thresholds, all checkpoints and final manifests.
2. Verify original v1/v2 artifact bytes remain intact. Run compatibility tests and old workflows against their preserved source/locks rather than pretending the new runtime is the old hash.
3. Run focused contracts, workload, capture, label-leakage, reward, masks, SARSA and DQN resume tests; then the required full regression before sealing the new package source.
4. Write a simple benchmark report with scenario coverage, all seed results, exact/pair metrics, score, probes, decision timing, training compute and limits.
5. Update the manual replay demo to load v3 cause sets and DQN definitions. Support `--policy dqn`, the two SARSA methods, VOI and DP, and print multiple roots explicitly.
6. Keep a new live manual capture workflow as a separate interface task if desired. Do not advertise a replay command as a continuously running live agent.
7. Verify no experiment controller remains and pause any implementation follow-up after the authorized plan completes. Save pause/resume checksums whenever the user requests a rest.

Exit check: a reviewer can trace every reported number to immutable data and frozen method definitions. Nothing requires the user to rerun an old reserved benchmark.

## 4. Proposed command sequence

These commands describe the interface to implement. They are **not available yet** and must not be pasted into the current v2 CLI expecting them to work.

```powershell
# Software and independent mechanics pilot.
rca-investigate validate-profile --config configs/investigation_v3_multi.yaml
rca-investigate validate-multicause-fixtures
rca-investigate verify-profile --config configs/investigation_v3_multi.yaml
rca-investigate capture-raw --config configs/investigation_v3_multi.yaml --schedule configs/capture_v3_pilot.yaml

# One independent evaluation schedule, then train-only fitting.
rca-investigate capture-raw --config configs/investigation_v3_multi.yaml --schedule configs/capture_v3_evaluation.yaml
rca-investigate split-raw --dataset datasets/raw_v3 --schedule-state runs/adaptive_v3_multi/capture/schedule.json
rca-investigate fit-encoder --dataset datasets/raw_v3 --split train
rca-investigate encode-dataset --config configs/investigation_v3_multi.yaml
rca-investigate fit-model --config configs/investigation_v3_multi.yaml --split train
rca-investigate full-evidence-reference --config configs/investigation_v3_multi.yaml --split validation

# Five seeds per learner, separate method directories, common task.
rca-investigate train --config configs/investigation_v3_multi.yaml --policy expected-sarsa --episodes 10000 --seeds 11 22 33 44 55
rca-investigate train --config configs/investigation_v3_multi.yaml --policy two-step-sarsa --episodes 10000 --seeds 11 22 33 44 55
rca-investigate train --config configs/investigation_v3_multi.yaml --policy dqn --episodes 10000 --seeds 11 22 33 44 55
rca-investigate compare --config configs/investigation_v3_multi.yaml --split validation
rca-investigate compare --config configs/investigation_v3_multi.yaml --split test
```

The controller resolves each method's environment and output path, refuses overlapping owners, checks saved identities and skips completed stages. Every command must have an explicit failure report and a dry-run mode where appropriate.

## 5. Resource estimate and practical sequence

For 440 evaluation captures, a protocol lower bound of 280 seconds each is about 34.2 hours before recovery, readiness and rejected attempts. At the previous observed pace of roughly ten to twelve accepted incidents per hour, collection alone is about 37 to 44 hours. Add mechanics pilots and rest periods. This is a planning estimate, not a promised completion time.

Allow roughly two to four working days for contracts, workload/component verification, multi-cause scoring and DQN implementation. Measure CPU training speed with a bounded fixture run before estimating the full DQN training duration. Do not promise GPU speed on this laptop or confuse small replay training with physical collection time. DP/hybrid comparisons may again take hours.

A reasonable first complete benchmark allowance is about five to eight calendar days including collection, checks and laptop rest. Broad additional sweeps increase that estimate. The most likely blockers are auth/fulfilment workload verification and fault masking, rather than neural-network size.

Build and verify the four single scenarios first. Add the composed pair controller next. Pass the multi-cause diagnostic gate before training. Compare all existing methods with separately declared DQN last. The aim is to measure whether DQN buys better evidence decisions, not to make a neural method win.
