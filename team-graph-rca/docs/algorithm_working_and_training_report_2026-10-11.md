# How the investigation methods work and how they were trained

11 October 2026, IST

This report explains four approaches used here: Expected SARSA, value of information or VOI, dynamic programming or DP, and hybrid SARSA. It also explains two-step SARSA, which is the second learning algorithm inside the tabular and hybrid benchmarks.

The explanations describe the implemented code. Results refer to the revised v2c main benchmark on 60 reserved incidents in ten sessions. Its challenges and evidence/cost sweeps are still pending. The original failed memory encoder remains preserved.

## 1. What problem are these methods solving?

The system investigates six possible conditions: catalogue database stopped, orders stopped, payment CPU contention, payment-path delay, carts memory pressure, and healthy operation.

At the start, it does not know the cause. It can request evidence such as status, CPU, memory, latency or logs. Each request costs credits. After receiving evidence, it must decide whether to request another tool, stop and give a diagnosis, or defer.

An investigation allows at most five probes and ten credits. Those are different limits: a probe can cost more than one credit. Repeated, unsupported and unaffordable actions are removed from the available action list.

The objective is:

`episode score = final-answer reward − 0.05 × credits spent`

A correct diagnosis earns 1, a wrong diagnosis earns 0, and deferral earns 0.25. For example, a correct diagnosis after four credits scores 0.80. Deferring after two credits scores 0.15. An unnecessary probe can lower the score even if the final answer is right.

Two separate components cooperate:

- The **diagnostic model** estimates which cause is likely from the evidence received.
- The **action-selection policy** decides what evidence to request and when to stop.

When a policy chooses STOP, the environment returns the diagnostic model's most likely cause. SARSA does not directly learn a separate fault classifier. It learns how to gather evidence for the shared classifier.

## 2. The diagnostic model shared by all methods

The fitted model stores 120 training examples. Each example contains its known cause and the categorical responses of all 14 tools. These are prototypes of what an incident looks like.

Initially, each of the six causes has probability 1/6. This is a balanced prior. As evidence arrives, the model gives more weight to training prototypes that agree with it, then adds together prototype weights belonging to each cause.

This follows Bayesian updating:

`updated belief ∝ prior belief × likelihood of the observed evidence`

Here, likelihood means how compatible an observation is with a training prototype. Smoothing of 0.1 gives mismatching observations a small probability rather than ruling them out completely. Several observations update the weights together. The model also predicts the possible response probabilities for a tool that has not yet been requested; planners use these predictions to evaluate future actions.

This is a finite prototype model, not a neural network or a perfect model of Docker. Its beliefs depend on the encoded categories and the examples collected. Confidence can be high and still be wrong if the current case differs from those examples.

The original memory encoder labelled ordinary validation memory around 58% as pressure. The revised encoder uses a 70.6% boundary fitted from training windows, between the highest healthy value and the lowest verified memory-fault value. The full-evidence validation check improved from 40/60 to 60/60 correct. That check supplies all evidence without the policy's acquisition limits, so it does not guarantee that a limited-probe policy will obtain the right evidence.

## 3. Expected SARSA: learn the value of each action

### The theoretical idea

Reinforcement learning learns decisions from rewards. A value called `Q(state, action)` estimates the total future score from choosing an action in a particular state and then following the policy.

A state here includes acquired evidence, remaining credits and probes, available actions, prices and experiment identities. Evidence history is sorted in the state key, so the same acquired evidence set can share a state even if requested in a different order. Incident IDs and the hidden true cause are not state inputs.

Expected SARSA uses this update:

`target = reward + gamma × sum over next actions [policy probability × Q(next state, next action)]`

`new Q = old Q + alpha × (target − old Q)`

At the end of an episode, the target is the terminal reward because there is no future action. The difference between target and old Q is the prediction error. Learning moves the estimate partway toward the observed target.

The word “expected” refers to averaging the next action values under the policy's probabilities. It does not mean the algorithm sees future evidence. Compared with bootstrapping from one sampled next action, this averages out that particular source of randomness. It remains affected by sampled incidents and evidence paths.

SARSA is an on-policy method: it estimates returns while accounting for the behavior policy used during learning, including exploration.

### How it works here

The implementation uses a dictionary of Q values, rather than a neural network. It initializes STOP using the model's current maximum diagnosis probability, DEFER at 0.25, and a new probe optimistically at `1 − its cost penalty`.

Training alternates useful-looking actions with exploration. Each probe produces its cost penalty immediately. At termination, a separate scorer adds the correct, incorrect or deferral reward. The hidden training label supplies this reward but is not passed to the policy as evidence.

Suppose a memory probe costs two credits. Its immediate reward is −0.10. If it helps obtain a correct diagnosis, that episode can score 0.90. Repeated training should raise its value in relevant states. If it rarely helps, repeated penalties should reduce its value.

During evaluation, exploration is zero and the policy chooses the largest learned Q value, with a stable tie break. It still consults the shared model when stopping. It is not guaranteed to find the best sequence after a finite number of episodes.

### What happened in v2c

Tabular Expected SARSA averaged 76.7% accuracy, score 0.501 and 3.90 probes. It chose actions quickly, about 0.185 ms per choice. Seed 11 missed all ten carts memory cases while correctly identifying the other four fault classes. This supports an evidence-selection weakness for memory cases, rather than a repeat of the original full-evidence encoder failure.

Its low computation cost is useful, but a fast choice is not automatically a useful choice. Increasing training may help, but the current results do not establish that it will close the gap.

## 4. Two-step SARSA: learn from two rewards before bootstrapping

This variant uses the same Q table, state representation, exploration schedule and rewards. The difference is its learning target:

`target = first reward + gamma × second reward + gamma² × Q(state two steps later, sampled action)`

It waits for two transitions before updating the earlier decision. If the investigation ends sooner, it uses the remaining observed rewards and does not bootstrap beyond termination. The code flushes these remaining updates at the episode end.

The theoretical tradeoff is credit assignment. Two observed rewards can carry the benefit of a later decision back to an earlier probe more directly. However, the target depends on sampled behavior and can be more variable. Two-step SARSA is therefore not automatically better than Expected SARSA.

In v2c, tabular two-step SARSA averaged 79.7% accuracy, score 0.529 and 3.72 probes. Its average computation was 0.129 ms per choice. That is better than tabular Expected SARSA in this saved benchmark, but five seeds on one cohort do not establish a universal advantage.

## 5. VOI: is one more probe worth paying for?

### The theoretical idea

Value of information compares the expected benefit of acquiring evidence against its cost. The objective is better decisions, not information for its own sake. A tool that reduces uncertainty can still be a poor purchase if it costs too much or does not change the final decision.

The value of stopping is the highest current diagnosis probability, under the 1/0 answer rewards. The value of deferral is 0.25. VOI considers each affordable, unused tool and each response the model predicts it could return.

For each tool, the implemented calculation is:

`probe value = −cost penalty + sum [response probability × best terminal value after that response]`

The terminal choice after the hypothetical response is STOP or DEFER. VOI chooses the highest-valued option among those probes and stopping or deferring now.

### How it works here

VOI is the shared bounded planner with depth 1. It looks ahead through one probe, then recalculates after the actual result. It can therefore use multiple probes during an investigation, although each individual calculation considers only one future probe.

For example, before seeing evidence, latency can split normal operation from slow operation and from cases where successful checkout timing is unavailable. After a normal result, memory may distinguish healthy operation from carts pressure. After a slow result, CPU may help distinguish payment contention from path delay. The exact choices follow model probabilities and prices.

VOI needs the fitted diagnostic model, but it does not train a Q table for 10,000 episodes. Its decisions are calculations from the current model.

### Result and limitation

VOI reached 93.3% accuracy, score 0.720 and 2.30 probes, at 27.59 ms per action choice. It correctly identified all 50 injected faults; its four errors were healthy cases diagnosed as payment or path problems.

One-probe lookahead can miss a useful sequence where the first probe only becomes valuable after a second probe. This is the reason to compare it with deeper planning.

## 6. DP: compare sequences of possible probes

### The theoretical idea

Dynamic programming evaluates a decision by combining its immediate reward with the value of the next decision. This is the Bellman principle: choose an action based on its cost and the best continuation after each possible outcome.

The recursive calculation is:

`value of probe = −cost penalty + sum [response probability × best value of the next state]`

Unlike VOI, the next state's value may include another probe before stopping. The search branches over possible observations, removes spent credits and probes, and prevents repeated evidence requests.

### How it works here

The main DP planner searches to depth 2, with at most 50,000 expanded search states per choice. It caches repeated search states within a choice to avoid recalculating them. It rebuilds the search after each actual observation.

If the expansion limit is exceeded, it falls back to depth 1. The reported DP fallback frequency was zero in this main test. This is bounded, two-probe planning, not a proof of global optimality across every five-probe investigation.

DP can recognize a sequence that starts with a weak-looking probe but leads to a valuable second probe. The cost is a larger search tree. More tools and response categories increase the work.

### Result and limitation

DP matched VOI: 93.3% accuracy, score 0.720 and 2.30 probes. Its mean decision computation was 1,663.50 ms, about 1.66 seconds per choice. VOI took about 27.59 ms, roughly 60 times less in this run. The extra lookahead provided no measured main-test score improvement.

Both planners depend on the same fitted model. A deeper search cannot repair misleading evidence or inaccurate predicted response probabilities by itself.

## 7. Hybrid SARSA: use learned values, with a planner fallback

Hybrid is an execution rule around a trained SARSA table. It is not a third SARSA update formula.

For each current state, the code totals its recorded training action visits:

- Fewer than five visits: use the depth-2 DP planner.
- Five or more visits: use the learned tabular policy.

The theoretical motivation is to trust cheap learned decisions where experience exists and spend computation where experience is sparse. However, visit count is only a rough reliability measure. Five visits do not prove that an action is correct or well learned. The fallback does not specifically detect classification uncertainty.

At evaluation, choosing a fallback action does not train the Q table. Training produced the tabular checkpoint; hybrid evaluation wraps that checkpoint with the planner.

Expected SARSA hybrid averaged 81.7% accuracy versus 76.7% tabular. Two-step hybrid averaged 88.3% versus 79.7% tabular. Their average fallback frequencies were 8.6% and 12.6% of action choices. Some seeds recovered every carts fault using fallback, but outcomes varied across seeds.

The computation tradeoff was substantial: Expected SARSA hybrid averaged 126.04 ms per choice, and two-step hybrid 89.20 ms. A small fraction of expensive planner calls can dominate average runtime. Hybrid improves these saved results but still uses more probes and achieves a lower score than the main planners.

## 8. Training and evaluation methodology

### Physical data collection

V2c collected 240 distinct accepted physical incidents across 40 capture sessions. Each session contains one case from each of six classes. Sessions, not individual rows, were assigned to train, validation or test:

| Partition | Sessions | Incidents | Purpose |
|---|---:|---:|---|
| Training | 20 | 120 | Fit encoder/model and update Q values |
| Validation | 10 | 60 | Check evidence and select checkpoints/policies |
| Reserved test | 10 | 60 | Evaluate frozen choices |

Grouping prevents cases from one capture session appearing on both sides of a split. Healthy controls include ordinary load, higher load, and temporary allocation/release. Accepted captures require measured effects and confirmed recovery. Saved identities and checksums make changes detectable.

### Encoder revision and fitting

The original run failed its diagnostic gate at 66.7%. It was preserved. The user authorized a separate memory revision using existing raw captures. The revision fitted its boundary from twenty healthy and twenty verified pressure training windows in independent sessions. No new faults were injected.

The revised dataset and model have their own identities. Other encoder values were retained. Validation reached 100%, passing the unchanged 90% gate. Because validation had already informed the decision to revise memory encoding, this validation success is development evidence; the later reserved comparison provides the separate held-out check. The current reserved results must not be used to retune this revision.

### Replay policy training

Each learner runs five seeds: 11, 22, 33, 44 and 55. Each seed completes 10,000 episodes. The main experiment therefore contains ten runs and 100,000 replay episodes.

An episode samples one of the 120 training incidents uniformly, with replacement. It creates a fresh investigation with an empty evidence ledger. Requesting a tool returns that incident's saved response. It does not execute a new physical Docker probe.

The policy sees public observations only. A separate scorer reads the true training label at termination and supplies the reward. Both learners use the same frozen model, data and settings:

| Setting | Value and meaning |
|---|---|
| Learning rate, alpha | 0.1; move 10% toward each target |
| Discount factor, gamma | 1.0; do not discount later episode rewards |
| Exploration, epsilon | Falls from 0.20 to 0.05 over 8,000 episodes, then stays at 0.05 |
| Validation interval | Every 500 episodes, including the final episode |
| Checkpoint interval | Every 50 episodes and at the stopping boundary |

Epsilon exploration mixes a uniform random action with greedy action selection. At epsilon 0.20, the random component has 20% weight; the actual probability of a greedy action also includes its share of that component. Tied greedy actions are sampled during training.

Ten thousand episodes repeatedly reuse saved incidents. They provide learning updates and exploration opportunities, not ten thousand independent physical replications or new fault patterns.

### Checkpoint selection and test protection

Validation evaluates a copy of the current policy, so it cannot change the training Q table or consume the training random generator. The best checkpoint is selected by validation mean return, with lower credits breaking a tie. The tested checkpoint may come from an earlier episode even though all 10,000 training episodes completed.

After all seeds finish, the validation comparison evaluates planners, tabular learners, hybrid learners and baselines. It freezes policy definitions, model/config/source identities and checkpoint hashes before opening the reserved test. Validation selected VOI for this main revision.

The evaluation uses zero exploration. The same reserved incidents are reused across methods for paired comparisons. Uncertainty comparisons group by capture session, so algorithms and seeds do not count as independent physical samples. Promotion also requires the declared accuracy and return-advantage conditions; no learner passed main v2c promotion.

## 9. Main revised v2c metrics

Learner and hybrid entries are arithmetic averages of their five per-seed summaries. Planner entries are single deterministic policy evaluations. Decision time includes terminal action choices, excludes tool waiting and training, and depends on machine load. It is not total investigation duration.

| Method | Accuracy | Score | Probes | Credits | Decision ms |
|---|---:|---:|---:|---:|---:|
| Expected SARSA, tabular | 76.7% | 0.501 | 3.90 | 5.32 | 0.185 |
| Two-step SARSA, tabular | 79.7% | 0.529 | 3.72 | 5.36 | 0.129 |
| Expected SARSA, hybrid | 81.7% | 0.556 | 3.65 | 5.21 | 126.04 |
| Two-step SARSA, hybrid | 88.3% | 0.609 | 3.56 | 5.49 | 89.20 |
| VOI, depth 1 | 93.3% | 0.720 | 2.30 | 4.27 | 27.59 |
| DP, depth 2 | 93.3% | 0.720 | 2.30 | 4.27 | 1,663.50 |

VOI and DP performed best on score and average probes. Tabular learners were cheapest computationally, but their evidence choices missed more cases and spent more credits. Hybrid recovered some accuracy through planning, with a substantial computation cost.

This does not contradict v2b, where learned policies matched DP. The larger v2c scope, memory evidence and healthy controls created a different task under the same 10,000-episode training allowance. The measured gap concerns this task and these settings. The remaining sweeps will test further conditions without changing the frozen main result.

## 10. Code and evidence behind this report

- `src/rca_investigation/belief.py`: prototype probabilities and train-only model fitting.
- `src/rca_investigation/environment.py`: observations, legal actions, rewards and stopping.
- `src/rca_investigation/policies/expected_sarsa.py`: Q table, exploration and hybrid visit rule.
- `src/rca_investigation/policies/n_step_sarsa.py`: Expected and two-step update targets.
- `src/rca_investigation/policies/dp.py`: bounded planning, caching and search-limit fallback.
- `src/rca_investigation/train.py`: sampling, validation selection and resumable checkpoints.
- `src/rca_investigation/comparison.py`: policy freezing and planner depths.
- `runs/adaptive_v2c_memory_separation_r1/main-comparison-verification.json`: verified main comparison.
- `runs/adaptive_v2c_memory_separation_r1/method-report-metrics.json`: numbers used in the table.

The interpretations above follow the implementation and saved results. They do not claim that the methods are optimal or that further training will necessarily improve them.
