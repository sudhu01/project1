# RL investigation agent: implementation review and plan

Prepared on 12 September 2026.

Reviewed source: [RL_RCA_Design_and_Methodology (1).md](<./RL_RCA_Design_and_Methodology%20(1).md>). This was the only project document in the folder. There was no implementation to inspect. The recommendations below account for your clarification that semantic tools will execute AWS CLI commands through Bash, replacing the abstract's proposed boto3 binding.

The report separates corrections to the abstract from implementation choices I recommend. Numerical rewards, budgets, network sizes, and training settings are starting configurations, not measured results. I have not run training, provisioned infrastructure, or executed AWS commands.

For the implementation decisions, start with [tools](#4-how-the-aws-cli-tools-should-work), [state and evidence](#5-state-space-and-evidence-processing), [rewards](#6-reward-formation-with-worked-examples), and [policy](#7-policy-architecture-and-algorithm). The build sequence is in [training](#9-the-actual-training-process) and [implementation order](#11-step-by-step-implementation-order-and-ownership).

## 1. My assessment

I would build this project, but I would reduce its first version substantially. The useful research question is whether a small learned policy can choose a sequence of investigations that reaches a reliable diagnosis at lower cost than sensible scripted and prompted alternatives.

The strongest part of the abstract is the division of responsibility. RL chooses the next probe and decides when to stop. A separate evidence processor interprets observations and maintains a diagnosis. This makes the learning problem small enough to study without training an entire language model.

The biggest risk is the environment you train in. PPO cannot compensate for a simulator that gives misleading evidence, exposes the answer through features, or makes every fault diagnosable through one obvious query. Most of your early work should define what the agent can observe, what each action reveals, and how you know the diagnosis is correct.

My recommended first system is:

| Decision | First implementation | Reason |
|---|---|---|
| Task | Investigate one known incident with one injected cause | Gives an unambiguous reward and keeps labeling manageable |
| Incident clock | Frozen telemetry snapshot with a declared cutoff | Makes repeated investigations comparable |
| Targets | A small graph of services and supporting resources | Enough structure to test routing without requiring a full knowledge graph platform |
| Actions | Tool, target, and a small approved parameter preset, plus STOP | Lets RL learn investigation choices and query scope |
| Evidence | Deterministic numerical features, with an LLM for unstructured text | Avoids paying an LLM to interpret ordinary metric arrays |
| Diagnosis | A frozen, calibrated structured belief estimator | Separates diagnosis errors from probe-selection errors |
| Policy | Masked categorical policy over valid action candidates | Simple probability distribution with explicit action feasibility |
| Encoder | Small shared feature network first, then a relational GNN | Establishes whether the graph network contributes |
| Learning | PPO with a value head | Fits short sequential episodes and a resettable simulator |
| Reward | Correct terminal diagnosis minus measured investigation cost | Aligns training with the actual deliverable |
| Training data | Small exact simulator, then realistic replay and simulator mixtures | Cheap debugging followed by a meaningful transfer test |
| First comparison | Random order, scripted investigation, and myopic value of information | Determines whether learning a sequence is worth the work |

I would postpone policy-controlled graph expansion, simultaneous faults, free-form query generation, and offline RL. Each is a separate source of difficulty. You can add them after the first agent shows a repeatable advantage.

## 2. What I would keep and what I would change

### Keep the policy's control of STOP

This is a good decision. Spending another minute or another query is an action choice. If the LLM decides when an investigation ends, you cannot cleanly attribute the cost savings to RL.

The evidence processor should update candidate scores. The policy should decide whether the expected benefit of more evidence justifies its cost. A hard deadline or budget still ends the episode even if the policy wants to continue.

### Keep the semantic tool interface, but include parameters in the action

The abstract uses a pair consisting of verb and node. Your proposed tools have adjustable parameters, so the real action is closer to:

```text
action = tool + target + parameter preset

Example:
metrics + payments + latency_and_errors_5m
logs    + payments + errors_5m
logs    + payments + errors_15m
```

Two calls to the same tool on the same resource can have different cost and information value. A policy that cannot choose between those scopes is missing a central part of your problem.

### Correct the Bayesian claims

An LLM-produced number called a likelihood ratio is not automatically a likelihood ratio. The proposed update is an additive update of log probabilities followed by normalization. It is not the usual binary log-odds update, and it is only Bayesian when its evidence terms have the right conditional likelihood interpretation.

Likewise, knowing how to sample telemetry from a simulator does not imply that its likelihood is tractable. Hidden propagation delays, noise, and shared latent causes may require integration over many possibilities. An exact updater is realistic for a deliberately small model. For the richer simulator, expect an approximate estimator.

### Correct the reward-shaping implementation

The abstract adds potential differences during probes but gives a separate accuracy-only terminal reward. In a finite episode, leaving out the potential correction at termination can change which policy is optimal. A policy may receive extra credit for becoming confidently wrong.

Start without shaping. If training needs it, implement the terminal correction described in Section 6. The original shaping theorem and later episodic analysis are relevant, but neither justifies omitting boundary conditions. [Ng, Harada, and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf), [Grzes on episodic reward shaping](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf).

### Treat the policy input as an approximate information state

Keeping raw LLM text out of backpropagation does not make the environment stationary. Changing the parser, its prompt, graph retrieval, telemetry distribution, or feature normalization still changes what the policy sees.

Also, a full-history MDP is a valid representation of a POMDP. Its practical problem is size. Your structured state compresses that history and may discard information. Describe it as an approximation unless you establish sufficiency for the chosen simulator.

### Do not assume complexity proves the need for RL

A large action space alone does not establish that RL will help. Nor does correlation between observations guarantee that one-step planning will fail. You need cases where a sequence has value that a short-sighted decision misses, followed by evidence that those cases matter on realistic incidents.

The sentence claiming deployment inspection is worthless before a later error-rate query is too strong. A recent relevant deployment may already be useful evidence. Test the value of combining probes instead of assuming it.

### Narrow the novelty claims

The claim that prompted planners cannot produce an accuracy-cost frontier is incorrect. Vary their budgets or stopping rules and they also produce operating points. ThinkFL explicitly discusses inference cost and latency, so avoid a broad claim that prior approaches ignore efficiency. Your defensible distinction is the small learned acquisition policy with a separately controlled evidence processor. [ThinkFL paper](https://arxiv.org/abs/2504.18776).

Sweeping a scalar cost penalty produces candidate tradeoff points. It does not guarantee discovery of the whole Pareto frontier, especially with imperfect training and non-convex achievable tradeoffs. Plot the observed non-dominated points and make claims about those points.

### Make the cost and benchmark claims conditional on evidence

Remove the unsupported universal ratio between log-query and metric-query prices. Measure representative workloads. One timing sample per action is inadequate when volume, pagination, retries, and LLM input size vary.

The cited RCAEval count of 735 cases and 11 fault types matches its published release. However, its suites have different telemetry coverage. PetShop is a metrics dataset, not a replay environment for every AWS command. Section 10 explains how that changes the experiment. [RCAEval release](https://zenodo.org/records/14504481), [PetShop repository](https://github.com/amazon-science/petshop-root-cause-analysis/).

The abstract also refers to six fault classes from an earlier abstract that is not in this folder. Those classes need an explicit definition. They should not remain an undocumented simulator assumption.

## 3. Define exactly what RL is learning

### 3.1 The task

An incident starts with an alert, a topology snapshot, an incident time window, and a budget. The agent can request selected telemetry. At the end, it returns a ranked list of suspected causes, a predicted fault family, and evidence references.

The agent learns an investigation strategy across many incidents. It does not learn by repeatedly modifying a production service until something improves.

An episode is one investigation. A step is one completed semantic probe or STOP. A probe may involve several AWS requests. Its internal polling is the tool executor's responsibility.

The policy does not receive the true root cause. During training, an evaluator holds that label separately and uses it to compute the terminal reward. In deployment, the policy runs without this reward signal. Later reviewed incident outcomes can become new training data.

### 3.2 Initial scope

Start with roughly 10 to 30 candidate resources and four supported fault families, subject to what the team can actually inject and observe. A reasonable provisional set is CPU saturation, memory exhaustion, network delay, and a bad application configuration or deployment. These are my proposed starting categories, not recovered definitions of the absent six classes.

Do not apply every fault family to every resource type. Maintain an applicability table. A database and an application service have different valid telemetry and failure mechanisms.

Choose one canonical labeling level. I would begin with service-level causes, including named backing services such as a database. Host-level faults can map to an affected service under a published rule for the first benchmark. If you later score infrastructure resources directly, change the label contract explicitly. A metric series or log stream is evidence about a cause, not a root cause entity.

Represent the candidate hypotheses as:

```text
H = all valid resource/fault-family pairs + UNKNOWN
```

UNKNOWN means the current candidate set or fault vocabulary may be incomplete. It is not a substitute for NO_FAULT. If false alarms become part of the task, add NO_FAULT with its own labeled episodes and scoring rule.

For the first closed-set training experiment, keep the true cause in the candidate graph and report that assumption. In a separate retrieval test, allow the cause to be outside it. Otherwise, a retrieval failure can quietly become a policy failure.

### 3.3 The hidden world and the visible information

The hidden world includes the cause, fault onset, workload, propagation, and the full telemetry record. The agent only sees the initial packet and observations it has acquired.

Freeze an evidence cutoff, such as alert time plus a fixed five-minute observation delay. All probe presets end at or before that cutoff. Use the same initial delay for every method and include it when reporting end-to-end time to diagnosis. A future repair event or a postmortem must never be visible during that investigation.

This first version studies sequential access to a fixed record. Later, a live version can advance time while queries run. That changes the transition model, introduces evidence freshness, and may require a WAIT or refresh action. Do not mix these two experiment definitions.

### 3.4 The objective

Use the following finite-episode objective:

```text
maximize expected diagnostic utility - lambda_cost * total investigation cost
subject to an episode budget and a maximum number of probes
```

The cost penalty expresses preference. The budget expresses a limit. They are different mechanisms.

For a first configuration, use at most 12 probes and a budget of 10 normalized cost units. Tune the eventual values to the measured tools. Start with one lambda_cost value, then train separate policies for a small sweep. A budget-conditioned policy is a later optimization, not a requirement for a first result.

If you train one policy across several cost penalties, supply lambda_cost as an input to both actor and critic and hold it fixed within each episode. Otherwise identical observations receive conflicting training targets with no way for the network to distinguish them.

## 4. How the AWS CLI tools should work

### 4.1 Use a registry of typed semantic tools

The policy should select an approved action object. A trusted executor resolves that object into AWS commands. Bash is the execution mechanism; the policy does not need to generate shell syntax.

Example action:

```json
{
  "tool": "logs",
  "target_id": "service:payments",
  "preset": "errors_5m",
  "incident_id": "incident-0042"
}
```

The registry resolves the target into an allowed account, region, log group, metric dimensions, or trace selector. It also resolves the incident window and preset into exact CLI parameters. IDs used for routing should not become learned embeddings that let the policy memorize a service's usual label.

For each tool, define input validation, applicable resource types, prerequisites, allowed presets, CLI implementation, output schema, cache identity, estimated cost, runtime limit, and failure behavior. Version this contract. All three backends must obey it.

Start with two useful presets per supported tool rather than an enormous cross-product of arbitrary settings. Parameters worth learning include time-window width, metric family, aggregation period, trace sample size, and approved log-query type. Account, credentials, endpoint, output format, retry limits, and maximum output size belong to the executor.

### 4.2 Map tools to real operations

| Semantic tool | AWS CLI binding | Practical condition |
|---|---|---|
| `metrics` | `aws cloudwatch get-metric-data` with a bounded query JSON | Resolve service-specific namespaces and dimensions. Batch a declared metric family rather than silently querying everything. |
| `logs` | `aws logs start-query`, bounded `get-query-results` polling, and `stop-query` when cancelling | One action owns the whole query lifecycle. |
| `traces` | `aws xray get-trace-summaries`, then `batch-get-traces` for selected IDs | Summary lookup and trace retrieval are separate calls. Respect sampling and missing traces. |
| `deploy_history` | ECS `list-service-deployments` and `describe-service-deployments`, or the deployment system actually used | ECS deployment history exists, but coverage and retention must match the incident. A current task definition alone is not the historical sequence. |
| `config_diff` | `aws configservice get-resource-config-history`, followed by a local structured diff | Requires recorded snapshots covering the resource and relevant times. No history is not proof of no change. |
| `resource_limits` | Resource-specific reads, such as ECS task definition CPU/memory limits | Distinguish configured limits, observed usage, and account quotas. These are different quantities. |
| `dependency_health` | A declared bundle of metrics or trace queries for selected neighbors | Publish its fan-out and charge every underlying read. Do not turn it into an all-system diagnosis tool. |

AWS documents metric batching, the two-stage trace retrieval flow, and ECS deployment-history operations. These establish the available APIs; the semantic grouping above is our design choice. [GetMetricData](https://docs.aws.amazon.com/cli/latest/reference/cloudwatch/get-metric-data.html), [X-Ray data retrieval](https://docs.aws.amazon.com/xray/latest/devguide/xray-api-gettingdata.html), [ECS list-service-deployments](https://docs.aws.amazon.com/cli/latest/reference/ecs/list-service-deployments.html), [ECS describe-service-deployments](https://docs.aws.amazon.com/cli/latest/reference/ecs/describe-service-deployments.html).

CloudTrail `lookup-events` can supplement change evidence, but it returns supported recent events within a region, including management events from the last 90 days. It is not a complete arbitrary application history. AWS Config returns recorded configuration items; the wrapper must choose and compare them. [CloudTrail lookup-events](https://docs.aws.amazon.com/cli/latest/reference/cloudtrail/lookup-events.html), [AWS Config history](https://docs.aws.amazon.com/cli/latest/reference/configservice/get-resource-config-history.html).

For new tracing instrumentation, use OpenTelemetry or AWS Distro for OpenTelemetry. AWS's current documentation places the X-Ray SDKs and daemon in maintenance mode. This concerns the instrumentation components, not withdrawal of the X-Ray service. [AWS support timeline](https://docs.aws.amazon.com/xray/latest/devguide/xray-sdk-daemon-timeline.html).

### 4.3 A concrete log-query flow

For `logs/payments/errors_5m`, the executor should:

1. Resolve payments to its registered log group and the incident cutoff.
2. Select a fixed query template and compute start/end times.
3. Write a validated request JSON into a private per-action working directory.
4. Start the query and record its query ID.
5. Poll under a shared request-rate limit and an action deadline.
6. Collect complete results, or record a partial/failed result explicitly.
7. Cancel unfinished server-side work when the deadline expires and record cancellation status.
8. Parse the result into the common observation schema and record the full cost.

Illustrative request JSON, with example values rather than a command to run against your account:

```json
{
  "logGroupName": "/project/payments",
  "startTime": 1789257300,
  "endTime": 1789257600,
  "queryString": "fields @timestamp, @message | filter @message like /timeout|refused|error/ | sort @timestamp desc | limit 50"
}
```

A fixed Bash worker can execute:

```bash
# The trusted wrapper creates this validated file in its private job directory.
# Region and credentials come from the worker's controlled environment.
aws logs start-query \
  --cli-input-json file://request.json \
  --output json \
  --no-cli-pager \
  --no-cli-auto-prompt
```

This snippet only illustrates query submission. The completed tool must implement polling, cancellation, parsing, and accounting. The Python process should invoke a fixed script with an argument list and `shell=False`; never concatenate model output into `bash -c` or use `eval`. Passing quoted values through a fixed script is compatible with your Bash requirement.

AWS returns a query ID from StartQuery. GetQueryResults can return partial results while the query runs and includes scan statistics. Use terminal status and `bytesScanned` in the wrapper. A small result limit bounds the returned text, not necessarily the amount of data scanned. [StartQuery](https://docs.aws.amazon.com/cli/latest/reference/logs/start-query.html), [GetQueryResults](https://docs.aws.amazon.com/cli/latest/reference/logs/get-query-results.html), [StopQuery](https://docs.aws.amazon.com/cli/latest/reference/logs/stop-query.html).

### 4.4 Return a common observation envelope

```json
{
  "schema_version": "1",
  "action_id": "step-0003",
  "status": "ok",
  "target_id": "service:payments",
  "window": {"start": "...", "end": "..."},
  "coverage": {"complete": true, "truncated": false},
  "evidence": [],
  "raw_artifact_id": "sha256:...",
  "cost": {
    "api_attempts": 4,
    "bytes_scanned": 12000000,
    "input_tokens": 420,
    "output_tokens": 90,
    "usd_estimate": 0.0,
    "latency_seconds": 2.4,
    "pricing_version": "region-and-date"
  }
}
```

The numbers are illustrative; zero in the example is not a claim that the operation is free. In actual records, use null for unknown cost components and distinguish provisional estimates from later billing reconciliation.

Use explicit statuses such as `ok`, `empty`, `unavailable`, `access_denied`, `timeout`, `throttled`, and `partial`. Empty evidence from a successful complete query differs from missing evidence. A timed-out log query should not make the resource look healthy.

The tool ledger should include retries, pages, polling, parser calls, and policy latency. AWS CLI pagination can issue multiple service requests automatically. Either manage pages explicitly or instrument them sufficiently to account for their cost. Do not equate one Bash invocation with one API request. [AWS CLI pagination](https://docs.aws.amazon.com/cli/latest/userguide/cli-usage-pagination.html).

### 4.5 Keep execution limits outside the learned policy

Give the worker credentials restricted to the required diagnostic reads and query lifecycle operations. Fault injection uses a separate identity controlled by the experiment runner. The policy should have no action that edits a service, changes IAM, or reads arbitrary local files.

Validate resource handles against the registry. Treat logs as data even when they contain text that looks like an instruction. An LLM's extracted evidence cannot add a new executable command to the registry.

Enforce deadlines, query scope, output limits, and account-level concurrency in the executor. These controls define the action space in which RL learns. A negative reward cannot substitute for them.

Mask exact successful duplicate queries in a frozen snapshot. Allow different presets when they can reveal additional evidence. Retry eligibility after a transient failure should follow a fixed bounded rule. A seven-bit visited mask is insufficient once parameters, overlapping windows, and freshness matter.

The simulator, replay loader, and AWS worker may share a schema without supporting every operation. A capability manifest must declare actual support. Missing logs in a metrics-only dataset should make the log action unavailable, not cause the adapter to invent a plausible log.

## 5. State space and evidence processing

### 5.1 What should enter the policy

The policy needs enough information to distinguish what is suspected, what has already been checked, what remains possible, and what another check would cost.

Use a graph, a table of node features, global features, and a table of candidate actions:

```text
policy_input = {
    graph_edges,
    node_features,
    global_features,
    candidate_actions,
    action_mask
}
```

Start with these node features:

| Group | Features | Why they matter |
|---|---|---|
| Identity category | Resource type and telemetry capabilities | Determines which probes make sense |
| Diagnosis | Candidate probability and conditional fault-family probabilities | Tells the policy where uncertainty remains |
| Acquired metrics | Robust baseline deviation, slope, error rate, latency, saturation | Describes measured symptoms |
| Missingness | A presence bit and age for each acquired feature | Prevents unknown values from looking normal |
| Structure | Directed hop distance from alert, degree, resource role | Helps distinguish the alerting caller from a failing dependency |
| Acquired change evidence | Deployment/configuration change flags and recency | Supports investigation of recent changes |
| Investigation history | Per-tool/preset status, covered windows, last result status, cost spent | Helps avoid redundant actions |
| Evidence quality | Coverage, contradictions, extraction failure flags | Makes unreliable observations visible |

Global features should include remaining budget in units and as a fraction, remaining probe count, diagnostic entropy, top-two probability gap, UNKNOWN mass, graph size, and the last action/status. Include time remaining if you enforce a time budget. Include the cost penalty if it varies between training episodes.

Each candidate action should carry a tool embedding, target index, preset features, expected cost, conservative cost estimate, expected latency, overlap with already observed evidence, and feasibility flag. These are estimates available before executing the action. The actual eventual scan cost belongs in the next observation, not the previous one.

Normalize continuous features with training-set statistics and record those statistics with the checkpoint. Use robust baselines for metrics whose variance can be nearly zero. Fit normal baselines from data available before the incident, not from the whole incident trace.

Do not make service names, fault-injection IDs, incident folder names, ground-truth labels, or simulator seeds predictive features. Randomly rename and reorder resources during training.

### 5.2 The expensive information cannot already be in the state

The abstract lists anomaly scores and deployment recency for every node. If the agent gets those values for free, it has already received much of the evidence that its tools are supposed to acquire.

Declare an initial information packet. I would allow the alert, static topology, cached capability metadata, and the metric that actually triggered the alert. All other incident-specific values remain unknown until a probe reveals them.

If the team wants a precomputed anomaly dashboard, that is also valid. Charge or report its collection cost, make it equally available to every baseline, and acknowledge that the task begins after that information has been acquired.

The same applies to GraphRAG. A retrieved historical incident with its known resolution can be useful prior evidence. A record describing the current test incident leaks its label. Split incident history by time and exclude current-incident descendants, reports, and duplicate runs before retrieval.

### 5.3 Use the LLM to extract facts, then update the diagnosis

I recommend splitting the abstract's updater into two components:

```text
raw result
  -> deterministic parsing and optional LLM extraction
  -> evidence ledger
  -> frozen diagnostic estimator
  -> belief vector
```

The LLM should extract facts such as a timeout involving a named dependency, an exception class, a changed configuration field, or a timestamped deployment event. It should attach an evidence reference. Do not require it to invent a precise probability for every hypothesis.

An extraction object might be:

```json
{
  "source": "action:step-0003",
  "facts": [
    {
      "subject": "service:payments",
      "relation": "connection_timeout_to",
      "object": "service:orders-db",
      "evidence_refs": ["row:17"],
      "observed_time": "..."
    }
  ],
  "coverage": "partial",
  "parse_status": "ok"
}
```

Only accept referenced facts whose subjects and objects can be mapped to the resource registry. Preserve unknown endpoints explicitly. A connection timeout means the caller observed a failure; it does not prove whether the caller, network, or database caused it.

Metric arithmetic, configuration diffs, and standard AWS JSON fields usually do not need an LLM. Route only text that benefits from interpretation through it. Freeze this routing rule across compared methods.

The diagnostic estimator can be a small shared scoring model over candidate resource/fault pairs. Train it on revealed-evidence sets using cause labels from training incidents. A cross-entropy loss teaches it to assign probability to the true hypothesis. Include partial histories, misleading symptoms, and failed probes. Calibrate its outputs on separate validation incidents, then freeze it before policy training.

This estimator is supervised. The action selector is reinforced by episode rewards. There is no requirement that every component use RL.

If data is initially too small for a learned estimator, use the exact toy-model posterior or a published rule-based scorer as the initial diagnostic component. Keep that component identical across the policy baselines.

### 5.4 If you retain the abstract's additive belief update

For hypotheses h, an exact update would be:

```text
b_next[h] = normalize(b[h] * P(observation | h, action, previous_history))
```

The conditional history matters. The second half of an overlapping log window is not an independent observation just because it arrived through a second action.

For an approximate score update, write it honestly as:

```text
b_next = softmax(log(b) + kappa * evidence_score)
```

Call the result an approximate belief. Clip and calibrate the evidence scores, but do not claim that clipping establishes Bayesian validity. Repeated small errors can still drive the distribution toward a wrong answer.

If the LLM sees the current prior, its response may already incorporate that prior. Adding the response as fresh likelihood evidence can count the prior twice. Asking only about the current top candidates also creates a feedback loop that can permanently exclude the true cause.

A better first design is to recompute the belief from a deduplicated evidence ledger using a frozen estimator. This allows the estimator to consider evidence combinations without treating each output as independent. It still needs calibration and can still be wrong, but its assumptions are easier to inspect.

Keep a small probability floor and UNKNOWN support. Use log-space normalization for numerical stability. Measure root-cause recall, negative log likelihood, Brier score, and reliability by confidence bin. Fit any temperature or trust coefficient on validation incidents, never the final test set.

### 5.5 Caching and updater drift

An action/observation cache key is sufficient only if those are the complete inputs to the cached computation. If the updater also sees the prior, top candidates, graph, or history, those inputs must be hashed too.

For context-independent fact extraction, cache by raw content hash, action semantics, resource mapping, schema version, prompt version, and model version. This produces better cache reuse than caching a prior-dependent posterior.

Do not use temperature zero as a reproducibility guarantee. Store actual outputs and validated fixtures. Test reproducibility with the stored outputs, and separately measure variation in uncached calls.

Separate two cost ledgers. The training ledger records what cached runs actually cost to execute. The evaluation ledger records the cost under the declared deployment cache policy. Replaying a cached LLM result during training does not make an unseen production observation free.

### 5.6 History and recurrence

Start with the explicit evidence ledger and coverage features. A graph feature vector plus the latest posterior may still lose order or temporal relationships. If paired experiments show that this matters, add a small recurrent memory over action/observation summaries.

A recurrent policy must carry memory during rollout and train on sequences with correct episode resets. Randomly shuffling individual transitions breaks that training setup. This is why I would first make the non-recurrent state as clear as possible.

## 6. Reward formation, with worked examples

### 6.1 Start with a simple terminal score

For the first experiment, define:

```text
U = 1 if the top-ranked canonical cause entity is correct, else 0
```

Use the marginal entity belief to produce that ranking. Predict the fault family conditional on the selected entity and evaluate it separately. This keeps the primary training objective aligned with AC@1.

If fault identification is part of the primary requirement later, a possible utility is:

```text
U = 0.8 * correct_entity + 0.2 * correct_entity_and_fault
```

The joint term prevents credit for guessing a common fault family on the wrong service. Changing utility also changes the optimal terminal decision rule. For this mixed utility, choose the entity/fault answer that maximizes expected utility under the belief, rather than assuming marginal top-1 is always optimal. Do not change the reward midway through a reported experiment.

Keep MRR and top-3 accuracy as evaluation metrics in version one. Giving MRR a large training weight can favor a broad ranking while weakening the top-1 result that operators care about. It is a legitimate alternative objective, but a different one.

Remove the abstract's undefined premature-stop penalty. A wrong early answer already loses the success reward. A separate penalty tied to the LLM's confidence could train obedience to an unreliable confidence threshold.

### 6.2 Define and measure cost

For a monetary-cost experiment, use:

```text
c_t = estimated incremental USD for the complete step / USD_reference
```

Choose USD_reference once from a calibration workload, for example the median cost of a designated standard probe including its evidence processing. The reference must be positive and frozen. It is only a unit conversion.

The incremental dollar estimate includes the tool requests, scanned data under the relevant billing rules, and any LLM processing. Record policy and parsing compute where material. Also report latency, tokens, bytes, and request attempts separately.

If latency belongs directly in the reward, use a declared extension:

```text
c_t = USD_t / USD_reference + w_time * seconds_t / seconds_reference
```

Then call the result a composite cost unit. A lower composite score does not automatically mean lower dollar spending or lower latency individually. Plot the raw measures too. Do not charge tokens again as money if their price is already included in USD, unless an extra context-use preference is intentional and disclosed.

CloudWatch pricing depends on the operation and usage dimension. Use current prices for the actual region, record a pricing version, and reconcile estimates against metered usage where possible. AWS responses do not universally return an exact dollar charge per request. [CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/).

Collect repeated samples for each preset across small and large telemetry volumes, cache conditions, and successful and failed calls. Fit cost estimates from observable properties such as query window, expected volume, and template. Store median and upper-quantile estimates. One fixed number per verb is not enough.

Report telemetry ingestion/storage and graph maintenance as standing system costs. Report initial incident setup as an episode cost. You may exclude a fixed setup cost from the action-dependent reward, but it must still appear in total cost comparisons. A method should not win by moving its probes into uncharged initialization.

### 6.3 The reward I would implement first

Use a finite horizon and gamma = 1. The explicit cost penalty already prices investigation length. A discount below one would add another preference for early terminal rewards, which would change the objective in the abstract.

```text
r_t = -lambda_cost * c_t + terminal_t * U_t
```

For a normal probe, terminal_t is zero. For STOP it is one. If a probe exhausts the episode budget or reaches the final permitted probe, evaluate the updated diagnosis and add U on that same transition. Do not require an unrecorded extra STOP, and do not pay U twice.

Budget exhaustion should still score the answer. Otherwise the policy may learn a quirk of termination handling instead of learning to investigate efficiently.

A failed probe pays its actual incurred cost and yields an explicit failure observation. Do not fabricate positive information reward for a successful API return, or a large punitive reward merely for a timeout. Feasibility validation belongs in the executor.

The agent learns through comparisons across episodes. If a useful probe tends to lead to correct diagnoses, the resulting terminal return makes that action more likely in similar information states. The critic helps assign that delayed return to earlier decisions.

### 6.4 A numerical example

Assume costs already expressed in normalized units and lambda_cost = 0.05. These values are illustrative.

| Investigation | Utility | Total cost | Return |
|---|---:|---:|---:|
| Correct after a cheap metric query and targeted logs | 1 | 4 | 0.80 |
| Correct after collecting nearly everything | 1 | 10 | 0.50 |
| Wrong after one cheap query | 0 | 1 | -0.05 |
| Stop immediately and happen to be correct | 1 | 0 | 1.00 |
| Stop immediately and be wrong | 0 | 0 | 0.00 |

The last two rows are intentional. An agent should not pay for evidence when the initial packet already supports the answer. Across uncertain incidents, however, guessing often earns less than investigating.

Suppose the current probability of a correct immediate answer is 0.55. A useful probe costs 2 units and raises expected final correctness to 0.80. Its expected utility gain is 0.25 and its penalty is 0.10, so it is worth taking under these assumptions.

If the current correctness probability is 0.95 and the same probe can only raise it to 0.98, the gain is 0.03. Stopping is preferable. These probabilities are for explanation; a learned policy estimates the value of actions from experience rather than receiving their true future outcomes.

This is the logic behind learned stopping. There is no universal confidence threshold that is optimal for every tool cost, incident, and remaining budget.

### 6.5 Budgets need their own enforcement

Before selecting an action, mask choices that cannot fit the remaining budget under the executor's reservation rule. After execution, charge actual incurred cost, release unused reservation, and update the budget.

For simulator credits and API-attempt limits, strict bounds are easy to enforce. Real log-query dollars can be uncertain before completion. A result-count limit or cancellation request is not a guaranteed hard spending cap. Use bounded windows, approved datasets, runtime limits, conservative reservations, and an experiment-wide cap. Record overruns rather than claiming that an estimated budget is mathematically strict.

If no probe is feasible, STOP remains available. A hard maximum of 12 probes prevents endless investigations even when the measured marginal dollar cost is close to zero.

Keep a diagnostic output such as `unresolved` for cases with weak or unsupported evidence. In the first experiment it receives no special success credit and does not replace the ranked output. If you later add a learned abstention action, define the cost of escalation and report coverage versus accuracy; otherwise always abstaining can become an easy strategy.

### 6.6 Add shaping only after the unshaped agent works

If sparse rewards make learning too slow, use a fixed potential over the information state x:

```text
Phi(x) = -beta * entropy(b(x)) / log(number_of_hypotheses)
Phi(terminal) = 0

r_shaped = r_base + gamma * Phi(x_next) - Phi(x)
```

Require at least two hypotheses or define the normalized entropy as zero for a singleton. Hold beta and the potential definition fixed within a training run. Start with a small beta such as 0.05 only as an experiment.

Apply the formula on every transition, including STOP and forced termination. For gamma = 1, the shaping terms telescope:

```text
sum of shaping terms = Phi(terminal) - Phi(initial) = -Phi(initial)
```

That total is independent of the chosen actions for a fixed initial condition. It can redistribute feedback without rewarding a different final objective. If the terminal potential remains belief-dependent, the leftover term can reward confidence itself. This is the boundary issue in the abstract. [Episodic shaping analysis](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf).

Entropy reduction can still provide misleading intermediate feedback when the belief is wrong. The terminal correction protects the return comparison under the stated conditions; it does not guarantee faster or stable neural-network training. Compare shaped and unshaped learning on the same incidents.

Do not use an LLM's self-rated informativeness as a reward. It is useful to log for analysis, but the policy should not be paid because another model said a query looked informative.

## 7. Policy architecture and algorithm

### 7.1 Use PPO for the first learned policy

Proximal Policy Optimization, or PPO, is an on-policy actor-critic method. The actor produces action probabilities. The critic predicts remaining return from the current information state. PPO repeatedly collects trajectories and limits their influence through a clipped training objective. It provides a practical starting point, not a guarantee of sample efficiency or a strict bound on every policy change. [PPO paper](https://arxiv.org/abs/1707.06347).

The environment has short episodes, discrete action candidates, explicit masks, and cheap resets in simulation. Those properties make PPO a sensible fit. There is no need to use an LLM fine-tuning algorithm merely because an LLM appears elsewhere in the system.

GRPO is not my fallback for this version. Your critic is a small network, not a second language model. Removing it is unlikely to remove the main expense, and group-based rollout comparisons may require more environment samples. ThinkFL's use of GRPO does not make it the natural algorithm for this different policy.

IQL is a separate offline-learning experiment, appropriate if you obtain sufficiently broad labeled trajectories. It cannot recover observations for actions absent from a fixed log. Poor coverage of tool choices and STOP decisions limits what it can learn. Do not treat it as a drop-in rescue when online PPO is expensive. [IQL paper](https://arxiv.org/abs/2110.06169).

### 7.2 Score each valid action candidate

Construct a table at each step:

```text
0  metrics  payments   latency_errors_5m
1  metrics  payments   saturation_5m
2  logs     payments   errors_5m
3  logs     payments   errors_15m
4  metrics  orders-db  saturation_5m
...
N  STOP
```

Use a shared network to score candidates. For action a targeting resource v:

```text
node_embedding[v] = encoder(graph, observed_node_features)[v]
graph_embedding = pool(node_embeddings)

score[a] = MLP(
    node_embedding[v],
    graph_embedding,
    global_features,
    tool_embedding[a],
    preset_features[a],
    estimated_cost_features[a]
)

stop_score = stop_MLP(graph_embedding, global_features)
action_probabilities = masked_softmax(all_scores)
value = critic_MLP(graph_embedding, global_features)
```

A joint candidate table is easier to debug than separate categorical heads for tool, node, and parameters. Independent heads can produce incompatible combinations. At larger scale, an autoregressive policy is possible, but it needs conditional masks and the correct joint action log probability.

During training, sample actions to explore. During primary evaluation, use a declared deterministic selection rule such as maximum score. Evaluate stochastic policies separately if needed. Do not switch modes invisibly when reporting results.

### 7.3 Start with a small encoder

First implement shared per-node MLP embeddings with pooled graph context and simple topology features. Then replace the encoder with two relation-aware message-passing layers, perhaps 64 to 128 hidden units. Add explicit reverse relation types so a caller and dependency can exchange evidence without losing direction.

A relation-aware GraphSAGE-style implementation is enough to start. Attention can be an ablation. PyTorch Geometric supplies heterogeneous graph utilities such as HeteroConv, so the team need not build message passing from scratch. [PyTorch Geometric heterogeneous graph documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html).

Keep the diagnostic estimator separate and frozen during the core policy experiment. The actor and critic may share their own encoder, but updating it must not silently change the beliefs that generated old PPO trajectories.

Shared scoring supports different graph sizes. It does not guarantee transfer to larger graphs or different topologies. Test that claim. Two layers also cannot directly encode arbitrarily long dependency chains; global pooling and hop features help, but they do not remove this limitation.

### 7.4 Masking and PPO bookkeeping

Mask unsupported tools, invalid parameter/target combinations, successful exact duplicates, and actions outside the budget rule. Keep STOP valid. A feasibility mask must depend only on information available to the deployed agent, not on the injected fault or whether a probe would be helpful.

Masking assigns zero probability to infeasible actions. Use the identical mask when sampling and when evaluating stored actions during PPO updates. This avoids a mismatch between the behavior distribution and the probability ratio used for training. [Invalid-action masking study](https://arxiv.org/abs/2006.14171).

Store the exact candidate table, mask, chosen candidate, old log probability, and old value with each transition. If candidate order changes later, a stored index alone may refer to a different action. Also snapshot observations rather than keeping references to mutable environment arrays.

For batches, pad candidate tables and graphs with explicit masks, or use graph batching plus per-graph candidate segments. Ensure padded actions cannot be sampled and padded nodes do not enter pooling. Log the number of valid actions because it affects policy entropy.

For the first implementation, use PyTorch, Gymnasium, and a compact PPO trainer adapted from a maintained reference. The custom action table and graph observation are the project-specific work. Pin dependencies after a smoke test; do not assume an off-the-shelf flat-action PPO class already handles all of these details.

## 8. Build the training environment before scaling the policy

### 8.1 Start with a small exact world

Build a diagnostic simulator with 5 to 10 resources, a few hypotheses, and discrete observations. Specify the probability of each observation for each hypothesis and probe. This is small enough to enumerate and check.

Sample one incident world at reset. Repeated access to the same record must return the same evidence. A probe of CPU and a probe of logs should both reflect that world's cause and workload. Do not sample unrelated causes or entirely fresh incident noise on every action.

For the simplest exact model, distinct probe outcomes can be conditionally independent given the hypothesis. Overlapping probes must share their underlying records. To model correlation exactly, add a small discrete latent workload or propagation variable, maintain a posterior over both cause and latent variable, and marginalize when producing cause probabilities.

Do not give the updater the sampled latent workload just because the simulator knows it. That produces an oracle with access to hidden truth, not an exact posterior based on acquired evidence. If exact inference becomes impractical, label the updater approximate and measure its error against the small exact cases.

This small environment should answer questions that are otherwise hard to debug in a neural agent:

| Test world | Expected behavior |
|---|---|
| A cheap probe reveals the cause reliably | Choose it and stop |
| Initial evidence identifies the cause | Stop immediately |
| All remaining probes are uninformative and costly | Stop |
| Two probes return the same evidence | Avoid buying the duplicate |
| A broad query costs more but can replace several narrow ones | Choose based on uncertainty and remaining budget |
| An expensive probe is the only useful discriminator | Buy it when the expected gain exceeds its cost |
| The same evidence arrives under shuffled resource IDs | Make the corresponding same choice |

Add a synthetic complementary-probe case. For example, a binary cause determines whether two observed bits agree, while either bit alone is uniformly random. Each individual probe has zero information about the cause, but together they identify it. This is a clean unit test for non-myopic acquisition, not a claim about cloud physics. Later test realistic evidence combinations such as deployment chronology plus affected dependency behavior.

For these tiny cases, enumerate short action sequences or use dynamic programming to find an optimal policy. Comparing PPO to a known optimum tells you whether the learning code works. Comparing only against random actions does not.

### 8.2 Build a replay environment

In replay, reset selects a labeled historical incident and hides its record. Actions reveal only supported slices of that record. A metrics preset selects the relevant resource, metric family, and available interval. A logs preset selects existing matching log rows. A trace preset returns existing trace records under a declared sampling rule.

This works for diagnostic reads because the action generally changes what the investigator knows rather than changing the historical service. It does not recreate the effect of a remediation action, and it cannot answer arbitrary commands unsupported by the stored data.

Use dataset-native granularity. If the data has five-minute samples, a one-minute preset should not create fictitious one-minute evidence. If configuration history is absent, mark that tool unsupported.

Replay's local file-read time is not AWS query time. Estimate deployment cost from the calibration model and report it as modeled cost. Keep real measured AWS results in a separate experiment. An identical tool schema makes adapters interoperable; it does not make their timing or observations physically identical.

### 8.3 Then build a richer simulator

Use healthy telemetry or captured fault-injection runs to inform the noise, volume, missingness, and propagation distributions. Randomize topology, workload, fault severity, onset, telemetry coverage, decoy anomalies, query failures, and action cost. Preserve coherent relations between these quantities.

The simulator should distinguish the dependency call direction from symptom propagation. A slow database can raise latency in an upstream caller. Host contention can affect several colocated services. A topology edge alone is not a causal proof, but it should constrain the kinds of propagation the simulator generates.

Keep some failure mechanisms and topology families out of training entirely. Randomizing parameters of one equation and testing new parameters of that same equation is weaker than testing a new propagation mechanism.

Audit cases where the true cause has little visible evidence. Some incidents will be intrinsically ambiguous under a budget. Do not add a perfect diagnostic marker merely to make every episode solvable. Report performance against the evidence that actually exists.

### 8.4 Make labels credible

For controlled runs, record the injected resource, mechanism, onset, and duration in a private evaluator manifest. Confirm that the injection caused a user-visible symptom and that it was not overshadowed by an unrelated background failure. Save failed or ambiguous injections separately rather than silently labeling them as clean examples.

Exclude chaos-controller messages, explicit injection labels, and answer-bearing file names from policy-visible data. Use fault recovery to validate the experiment externally, but do not let post-cutoff recovery evidence enter the agent's investigation.

Keep all windows and repeated runs from the same incident family or injection campaign in the same split when they share distinctive fingerprints. Otherwise train and test can contain nearly identical cases.

## 9. The actual training process

### 9.1 Prepare four separate data partitions

Use training incidents, updater-calibration incidents, policy-validation incidents, and a final test set. If data is scarce, cross-validation can replace fixed development partitions, but the final test decisions must remain untouched.

Generate simulator training incidents from one set of seeds and development/test incidents from disjoint seeds. Also hold out topology templates and mechanisms where possible. Seed separation alone does not prevent all structural overlap.

Train the evidence estimator on partial investigations generated by a mixture of random, scripted, and information-seeking policies. Calibrate it separately. Freeze it before core policy training. Later, collect additional histories visited by the learned policy, retrain the estimator only on training incidents, and start a clearly versioned policy-training phase. This addresses the different observations a stronger policy may seek.

### 9.2 Establish the diagnostic ceiling and simple baselines

Before a large PPO run, evaluate:

1. Immediate STOP using the initial diagnosis.
2. Random valid probe order, evaluated at several fixed stopping budgets.
3. A simple investigation script using the same probes and evidence processor.
4. A myopic value-of-information policy.
5. Full supported evidence passed through the same diagnostic estimator.

The fifth comparison measures the estimator's capability with available data. Treat it as a reference, not a mathematical upper bound: an imperfect estimator can be confused by extra noisy evidence. In the exact toy model, the full-information Bayes solution does provide the appropriate ideal reference.

For one-step value of information, match the task's utility:

```text
VOI(action) = expected best terminal utility after action
              - best terminal utility now
              - lambda_cost * expected action cost
```

Choose the highest positive value or stop. For entity accuracy and an exact belief, best terminal utility is the largest marginal entity probability. Greedy entropy reduction is a related comparator, but it optimizes uncertainty rather than the actual diagnosis utility. Include it separately if useful.

Exact VOI is available in the small simulator. For realistic data it requires an outcome model; label the approximation and account for its computation. A clairvoyant baseline may use simulator-only likelihoods as an oracle reference, but it must not be presented as an equally deployable competitor.

If cheap scripted or random investigations already perform well, look at the accuracy-cost curves. The next step may be better stopping or fewer tools, not a bigger policy network. If full evidence fails, fix diagnosis quality before blaming action selection.

### 9.3 Optional imitation warm-up

Generate successful teacher trajectories in the training simulator using a short-horizon planner or the best simple policy. Train the actor to predict the teacher's selected action from its visible information state and feasible candidate list.

A few thousand short trajectories are a reasonable pilot, subject to available coverage. Include correct STOP examples and several budget conditions. Avoid teaching only long successful investigations; that creates a bias against early stopping.

This warm-up is supervised imitation. It initializes sensible action preferences before PPO exploration. It is optional, and the final comparison should disclose whether it was used. A teacher is not automatically optimal, so PPO should be allowed to improve on it.

### 9.4 A starting PPO configuration

The following is a proposed configuration for modules the team will implement. It is not an already working command or configuration in this folder.

```yaml
environment:
  backend: simulator
  frozen_snapshot: true
  max_probes: 12
  budget_units: 10.0
  candidate_nodes_min: 10
  candidate_nodes_max: 30
  updater_version: frozen_v1

reward:
  terminal_utility: entity_top1
  lambda_cost: 0.05
  gamma: 1.0
  shaping_beta: 0.0

policy:
  encoder: relational_graph
  layers: 2
  hidden_dim: 128
  action_head: shared_candidate_scorer
  recurrent: false

ppo:
  parallel_envs: 16
  rollout_steps_per_env: 128
  minibatch_size: 256
  update_epochs: 4
  learning_rate: 0.0003
  clip_ratio: 0.2
  gae_lambda: 0.95
  value_loss_coefficient: 0.5
  entropy_coefficient_start: 0.01
  entropy_coefficient_end: 0.001
  max_gradient_norm: 0.5
  target_kl: 0.02

evaluation:
  every_updates: 5
  fixed_validation_incidents: 100
  deterministic_actions: true
```

These settings produce 2,048 transitions per collection batch. Four update epochs reuse that batch before fresh data collection. Run a pilot of roughly 100,000 transitions, inspect learning, and scale only if the curves justify it. There is no universal requirement that PPO use one million or ten million steps.

Use gamma for the task's discount and gae_lambda for the advantage estimator. Neither is lambda_cost. Name them separately in code.

The small graph network should not require a multi-GPU language-model training setup. Start by benchmarking on the available workstation. A GPU can accelerate network updates, but simulator throughput and parser calls may dominate. Measure before reserving hardware.

### 9.5 What happens in one PPO update

1. Freeze the current actor for collection and reset the parallel environments as episodes end.
2. Build each visible state and its candidate action table.
3. Sample a feasible action, execute the simulator tool, update evidence, and receive the reward.
4. Store the observation snapshot, mask, candidate mapping, action, reward, termination status, old log probability, and value estimate.
5. At the end of the rollout batch, compute returns and advantages.
6. Update the actor and critic over minibatches using PPO's clipped objective, value loss, and a small exploration entropy bonus.
7. Discard that rollout batch after the configured updates and collect fresh trajectories.

Generalized advantage estimation, or GAE, combines prediction errors over several future steps to estimate how much better an action turned out than the critic expected. Its decay parameter trades variance against bias. [GAE paper](https://arxiv.org/abs/1506.02438).

The temporal-difference error compares observed reward plus estimated remaining value with the critic's previous prediction:

```text
delta_t = reward_t + gamma * bootstrap_value_next - value_t
advantage_t = delta_t + gamma * gae_lambda * continuation_t * advantage_next
```

At a genuine terminal state, bootstrap_value_next is zero. At a rollout boundary that merely interrupts collection, bootstrap from the final visible state. Do not continue the advantage recursion into a new episode.

An investigation budget or probe horizon is part of the task, so reaching it is a terminal outcome with a scored diagnosis. A trainer's arbitrary collection cutoff is different. Include remaining time or steps in the state for the finite-horizon task. Gymnasium documents the distinction between termination and external truncation. [Gymnasium time-limit handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

Normalize advantages over valid transitions. Track approximate KL divergence and stop the current set of PPO update epochs early if it exceeds the configured threshold. Avoid normalizing rewards differently for different cost-penalty experiments, since that can complicate interpretation of their tradeoffs.

### 9.6 Environment pseudocode

This is the intended logic, not a tested implementation:

```python
def step(action):
    assert action in current_feasible_actions()
    old_info = snapshot_visible_information()
    old_potential = potential(old_info)

    if action.is_stop:
        step_cost = finalize_cost_if_any()
        terminal = True
    else:
        result = backend.execute(action)
        step_cost = result.complete_normalized_cost
        evidence_ledger.merge_deduplicated(result)
        belief = frozen_updater(evidence_ledger, visible_graph)
        spend += step_cost
        probe_count += 1
        terminal = budget_or_horizon_exhausted()

    next_info = snapshot_visible_information()
    reward = -lambda_cost * step_cost

    if terminal:
        prediction = decode_diagnosis(next_info)
        reward += evaluator.score(prediction)  # Labels stay private here.
        next_potential = 0.0
    else:
        next_potential = potential(next_info)

    reward += gamma * next_potential - old_potential
    return next_info, reward, terminal, audit_record_without_labels()
```

The real environment also needs reset logic, reservations, observation validation, and correct handling of asynchronous worker failures. Do not put ground truth in `info` fields that a policy wrapper might consume. Do not feed previous training reward into a recurrent actor when that reward contains label-derived information unavailable during deployment.

### 9.7 Measure learning, not just return

At each validation point, record entity accuracy, cost, latency, probe count, STOP rate, per-tool usage, invalid actions, budget overruns, updater calibration, and performance by fault family. Also log actor entropy, critic error or explained variance, KL divergence, and PPO clipping frequency.

Save several full investigation traces. You should be able to read what the agent knew, what it bought next, what changed in the diagnosis, and why the reward differed from a cheaper alternative. A rising scalar return can hide a collapse into immediate guessing when the cost penalty is too high.

Select checkpoints using a predeclared validation objective, such as highest accuracy within a fixed cost budget. Do not pick the final checkpoint by test performance or by whichever metric makes that run look best.

### 9.8 Transfer to realistic evidence without an LLM on every simulator step

Use this progression:

1. Debug policy learning with the exact small updater and no LLM calls.
2. Train the policy against the frozen approximate estimator using replay and realistic simulation.
3. Collect a bounded set of text observations from training incidents, run the chosen LLM extractor, review a sample, and cache its validated outputs.
4. Train a surrogate extractor if the corpus is large enough. Evaluate fact accuracy and downstream diagnostic error, not just agreement with the LLM.
5. Train a policy against the actual planned deployment evidence path, using cached outputs where valid. Keep a mixture of earlier training incidents to limit forgetting.
6. Evaluate on unseen incidents with the real extractor. Freeze all components before the final comparison.

The ideal-updater-to-real-updater gap is not just a matter of adding random noise to a scalar confidence. Real mistakes may consistently blame callers, omit minority candidates, or misread timeout messages. Include measured error patterns and failed extractions in training.

Be precise about transfer terminology. If you train the policy on target-system incidents, its evaluation on new incidents is target-domain adaptation. If you calibrate on target labels, it is not a fully untouched target system. Reserve zero-shot for the components and splits that actually receive no target adaptation.

### 9.9 Budget training from measured throughput

First time 10,000 simulator steps and several representative parser calls. Estimate:

```text
simulation hours = total transitions / measured transitions per second / 3600
LLM spend = uncached observations * average input/output token price per observation
```

For example, 10,000 episodes at 12 probes per episode could require up to 120,000 extraction opportunities. Caching helps only when the inputs repeat. The abstract's proposed thousands of episodes with the real updater may still be expensive.

After one successful penalty setting, a starting sweep is lambda_cost in {0, 0.01, 0.03, 0.1, 0.3}, adjusted if calibration puts tool costs on a different scale. Five independent seeds across five penalties means 25 runs. If each uses one million transitions, that is 25 million transitions before ablations. Plan that multiplication before committing to a large sweep.

Use a few seeds during debugging and five independent training seeds for the main claims if feasible. Save RNG states, optimizer state, model weights, dependency lockfile, simulator version, data split manifest, cost model, feature normalization, tool registry, and updater version with every checkpoint.

### 9.10 Increase difficulty when the current stage is understood

Move from small clean single-fault graphs to decoys, missing evidence, larger graphs, and tighter budgets. Keep some earlier cases in later training batches so the policy does not forget simple behavior.

A plateau alone is a poor promotion rule. An agent can plateau at bad performance because of a bug or insufficient evidence. Advance when its results approach the relevant toy optimum or reference baseline, its investigation traces make sense, and the next stage tests a specific limitation. Reserve simultaneous independent faults for an explicit extension.

## 10. Evaluation that can support the project's claim

### 10.1 Use benchmarks for what they contain

| Data source | Useful test | Limitation to disclose |
|---|---|---|
| Small exact simulator | Reward, stopping, masking, and non-myopic behavior | Constructed diagnostic tasks do not establish cloud realism |
| Rich simulator | Generalization across generated faults, graphs, and budgets | Results depend on the generative assumptions |
| RCAEval | Service-level localization using supported metrics, logs, and traces | Telemetry coverage differs across suites; replay presets must match available data |
| PetShop | Metric acquisition across AWS-shaped dependencies | It does not supply a complete CLI response history for all proposed tools |
| OpenRCA | Diagnosis with larger heterogeneous telemetry and system-specific labels | Requires explicit adaptation of target granularity and output scoring |
| Controlled cloud application | Real command behavior, latency, scan volume, cost, and transfer | Fewer episodes and narrower system coverage |

RCAEval's published release describes 735 cases from three systems and 11 fault types, with metrics-only and multimodal suites. Pin the chosen release and report the subset you actually use. [RCAEval dataset](https://huggingface.co/datasets/phamquiluan/RCAEval).

PetShop's official repository describes 68 injected performance issues and metrics from 41 components. Use it as a metric-based transfer test. Its repository is archived, so pin its data and code rather than expecting continuing compatibility updates. [PetShop repository](https://github.com/amazon-science/petshop-root-cause-analysis/).

OpenRCA's original evaluation includes task-specific root-cause elements, and its project now distinguishes releases. Select a release, inspect its evaluator, and state whether you report native scoring or your own mapped entity score. Do not compare mapped service top-1 against someone else's native multi-element metric. [OpenRCA repository](https://github.com/microsoft/OpenRCA), [OpenRCA project](https://microsoft.github.io/OpenRCA/).

### 10.2 Give every method a fair operating curve

Compare the learned policy against immediate STOP, random order at several budgets, the simple investigation script, myopic VOI, and a cost-aware prompted planner. Keep the initial packet, tool catalog, evidence cutoff, available telemetry, and cost accounting consistent.

Run the non-learned policies with the same evidence processor to isolate action selection. Then run a separate end-to-end LLM planner comparison that includes its planning tokens and model latency. That second comparison measures whole systems, not just policy quality.

For classical RCA, select a small reproducible set supported by the dataset rather than promising every method in the abstract. Charge or declare the complete telemetry that these methods require. A batch algorithm receiving all metrics is a useful full-data comparator, but it does not have the same acquisition task as a sequential agent.

Sweep baseline budgets and stopping thresholds too. A baseline should not be represented by one weak operating point while the learned policy gets a full sweep. Report accuracy at matched monetary cost and matched time where possible. Equal action counts are a secondary comparison because one semantic action can be much more expensive than another.

### 10.3 The primary result

The result I would aim to demonstrate is higher entity accuracy at a fixed acquisition budget, or lower cost at matched accuracy, on held-out incidents. Put uncertainty around that comparison.

Report:

| Measure | Interpretation |
|---|---|
| AC@1 | Whether the top entity is correct |
| AC@3 and MRR | AC@3 checks the first three entities. MRR averages 1 divided by the true entity's rank, with zero if it is absent. |
| Joint entity/fault accuracy | Whether the agent identifies both location and mechanism |
| Cost and latency distributions | What diagnosis costs, including expensive tail cases |
| Budget-overrun rate | Whether estimated reservations fail in practice |
| UNKNOWN or unresolved rate | How often the agent lacks a supported diagnosis |
| Retrieval recall | How often the true cause was even available as a candidate |
| Calibration | Whether confidence corresponds to observed correctness |

Use time to diagnosis or time to root-cause identification. Do not call this repair time because the agent is not repairing anything.

Plot all measured points and identify the observed non-dominated set. A policy with a higher cost penalty may still perform inconsistently because training is imperfect; do not force a monotone curve through it. Include both per-system results and clearly labeled macro averages. A pooled result can be supplementary, not the only conclusion.

Use paired incident comparisons because every method investigates the same cases. For confidence intervals, account for related incidents and repeated training seeds, for example by clustering incidents by injection campaign and reporting seed variability separately or using a hierarchical bootstrap. Five runs do not create five independent copies of every test incident.

### 10.4 The ablations I would prioritize

1. Learned policy versus random/scripted policies with the same diagnosis component.
2. Learned policy versus one-step utility-based VOI.
3. GNN encoder versus a parameter-matched shared MLP encoder.
4. Learned STOP versus fixed probe budgets and a calibrated threshold rule.
5. Adjustable presets versus one fixed preset per tool.
6. Realistic approximate updater versus the exact toy updater, only where an exact comparison exists.
7. No shaping versus correctly implemented shaping.
8. No cost term versus several nonzero penalties.

The parameter-preset ablation deserves a place in your project because adjustable CLI scope is part of the intended contribution. A graph policy that merely selects a resource while every query has a fixed scope tests a smaller claim.

Test ground-truth and feature isolation as well. Change the private label while holding visible observations constant; the actor input, mask, and unscored output must remain unchanged. Change only resource order; probabilities should map to the same semantic actions within numerical tolerance.

## 11. Step-by-step implementation order and ownership

### Step 1. Agree on an incident contract

Write one JSON schema defining the incident ID, alert, evidence cutoff, topology snapshot, candidate labels, telemetry availability, and hidden evaluator label. Decide the initial fault families and label granularity with the rest of the team.

Your output is a resettable episode specification. The telemetry/graph teammate supplies the resource mapping and recorded data. You own the visible/hidden boundary and evaluation rules.

Done means a second person can explain exactly what is available before the first action and what qualifies as a correct answer.

### Step 2. Implement two complete tools

Start with metrics and logs for one service type. Include two bounded presets, validation, common results, timeouts, error handling, and cost records. Capture successful, empty, partial, and failed examples.

Your output is the semantic action contract and candidate generation. The executor owner implements the fixed Bash scripts and AWS mappings. Agree on cost semantics together.

Done means the simulator and replay adapters can return the same schema, and the AWS worker can execute approved read probes in a controlled account. You do not need seven tools to validate the RL loop.

### Step 3. Build the small environment and evaluator

Implement reset, step, STOP, budgets, hidden scoring, and the exact diagnostic fixtures. Create cases with known optimal actions and termination choices. Make deterministic replay possible from a manifest.

You own this component. Done means the reward arithmetic and action masking pass meaningful tests before any neural training starts.

### Step 4. Establish baselines

Run immediate STOP, random order across budgets, a script, myopic VOI, and the exact toy optimum. Produce the first accuracy-cost plot and a few readable investigation traces.

Done means there is a measurable gap for a better sequential policy to address, or a documented reason to simplify the project's claim. Do this before investing in a large synthetic cloud model.

### Step 5. Train a small policy

Implement the shared candidate scorer and PPO. Start with the simple encoder, then compare the graph encoder. Run a short pilot with a frozen updater and a single penalty. Confirm it can overfit a tiny set as a debugging test, then evaluate on fresh incidents.

Done means it approaches known toy behavior and beats at least the weak baselines on held-out cases. A failure here calls for inspecting transitions, masks, and rewards before increasing training volume.

### Step 6. Build the realistic evidence path

Add the evidence ledger, deterministic parsers, LLM extraction, and a calibrated diagnostic estimator. Build replay for one data suite. Record capability gaps instead of hiding them with synthetic observations.

You own the estimator-to-policy interface and the transfer measurements. The LLM teammate can own extraction quality, prompts, and cache generation. Freeze agreed versions for comparisons.

Done means you can attribute failures to missing evidence, extraction, diagnosis, or action selection by inspecting a stored trace.

### Step 7. Expand tools and training difficulty

Add traces and change history when the underlying data supports them. Increase graph sizes and missingness. Test preset selection and train a small cost-penalty sweep. Keep development decisions off the test set.

Done means the policy's advantage survives a more realistic estimator and harder held-out incidents. If it disappears, that is the issue to solve before adding graph expansion or more neural layers.

### Step 8. Run controlled AWS validation

Use a small instrumented application and a separate fault-injection runner. Confirm that telemetry is ready before starting an investigation, but do not expose the injection record. Freeze the policy during these runs.

If the team uses Chaos Mesh, it needs a compatible Kubernetes environment. It is not a direct ECS fault-injection tool. Choose the injection mechanism after choosing the deployment platform, and map it to the fault definitions in the incident contract. [Chaos Mesh project](https://github.com/chaos-mesh/chaos-mesh).

Done means complete traces show real CLI execution, actual failure handling, and measured costs. Compare the same policy's replay result to its live result to quantify adapter and timing differences.

### Step 9. Freeze and evaluate

Freeze schemas, data splits, cost models, feature normalization, updaters, prompts, and checkpoints. Run the agreed seed set and baselines. Report failures by system and tool, not just the best aggregate.

Your final deliverable should contain a checkpoint, reproducible runner, incident manifests, evaluation script, cost ledger, and evidence-linked diagnoses.

### A suggested repository layout

The following layout is proposed. These modules do not yet exist in the folder.

```text
rl_rca/
  contracts/
    incident.py
    action.py
    observation.py
  tools/
    registry.py
    candidate_builder.py
    aws_cli_backend.py
    replay_backend.py
    simulator_backend.py
    scripts/
      metrics.sh
      logs.sh
  evidence/
    ledger.py
    parsers.py
    llm_extractor.py
    belief_estimator.py
    calibration.py
  env/
    investigation_env.py
    budget.py
    reward.py
  models/
    graph_encoder.py
    candidate_policy.py
  training/
    rollout.py
    ppo.py
    train.py
  evaluation/
    baselines.py
    metrics.py
    evaluate.py
  configs/
  tests/
  artifacts/
```

For the local development environment, use a pinned Python environment and a Linux worker/container or WSL for Bash execution. Pin the AWS CLI version independently from the Python dependencies. Keep credentials outside checkpoints and incident artifacts. Begin on CPU if necessary; benchmark before making GPU availability a dependency.

Once the modules exist, the intended workflow can look like this:

```bash
# Proposed entry points, not commands implemented by this report.
python -m rl_rca.evaluation.evaluate --config configs/toy_baselines.yaml
python -m rl_rca.training.train --config configs/ppo_toy.yaml
python -m rl_rca.training.train --config configs/ppo_replay.yaml
python -m rl_rca.evaluation.evaluate --config configs/heldout.yaml
```

### Tests worth writing

These tests protect the validity of the experiment rather than merely checking that functions return something:

- STOP and forced termination each score the diagnosis once.
- A failed action consumes incurred cost without inventing healthy evidence.
- A repeated observation cannot multiply its diagnostic support.
- The shaped and unshaped return difference is constant for alternative trajectories with the same initial state.
- Ground truth cannot alter actor inputs, capability masks, or pre-action costs.
- Candidate reordering preserves semantic action probabilities.
- A PPO update uses the same action support as its stored rollout.
- Replay never returns evidence after the incident cutoff.
- The worker cannot execute a command outside the registered templates.
- Identical fixture requests produce equivalent semantic results across supported adapters.

## 12. What to add after the first result

### Graph expansion

Start with deterministic subgraph retrieval or the whole small topology. Keep symptom-bearing edge weights hidden until acquired, or include their collection in the initial packet. Measure retrieval recall before claiming an investigation failure.

When adding EXPAND, charge for retrieval and any newly acquired telemetry. Newly discovered nodes should start with unknown dynamic features unless the expansion action explicitly bought that evidence.

Preserve probability mass for undiscovered candidates. One simple representation keeps UNKNOWN mass and transfers a calibrated fraction of it to newly introduced candidates using a prior over their resource types, leaving residual UNKNOWN mass. Do not reinitialize the entire belief every time the graph grows or silently remove uncertainty by renormalizing only visible nodes.

The belief and history may not be sufficient to allocate that mass accurately. Treat the rule as part of the estimator and validate it. There is no general reason expansion should happen late. A policy may need to expand immediately when the alert has a likely upstream dependency outside the retrieved graph.

### Multiple causes

A categorical distribution over one cause cannot represent two independent causes correctly. Normalizing all node probabilities to sum to one forces the hypotheses to compete.

For an extension, predict a set using a multilabel estimator or a structured distribution with a limit on set size. Specify how causes interact and use set-based evaluation. Decide how the reward balances missing a cause against reporting extra causes. This changes the task, so keep the single-cause result separate.

### Continuous parameters and concurrent probes

If discrete presets leave substantial performance unused, add more levels of scope before attempting unrestricted query generation. Continuous timestamps, SQL, and arbitrary shell text introduce validity and credit-assignment problems that your current research question does not require.

Concurrent probes can reduce wall-clock time while increasing spend and redundancy. They also require state for in-flight jobs and a different scheduling policy. Finish the sequential baseline first. Its trace logs will tell you whether concurrency would help.

### Offline learning from human investigations

If genuine investigation traces become available, record the information visible at each decision, all chosen parameters, outcomes, costs, final labels, and the reason the investigation ended. Such logs are more useful than a list of shell commands with no context.

Begin by testing imitation and coverage. Then decide whether an offline RL method has enough action support to improve on the recorded behavior. Human logs may contain actions your semantic registry cannot express; identify those gaps rather than discarding them silently.

## 13. The first decisions I would bring to the team

Before writing a large policy, settle these items in the incident and tool contracts:

| Decision | My default recommendation |
|---|---|
| Deployment target | One known platform first, chosen by the infrastructure team |
| Label level | Service-level canonical entity, with fault family scored separately |
| Supported faults | Four observable, reproducibly injected families before expanding |
| First tools | Metrics and logs, then traces and change history |
| Evidence clock | Frozen cutoff for the first experiment |
| Initial information | Alert plus static topology and explicitly declared cached metadata |
| Policy reward | Entity top-1 success minus normalized monetary cost |
| Operational limits | Probe count and time limits, with conservative cost reservations |
| Evidence processor | Frozen structured estimator; LLM used for text extraction |
| First learning algorithm | PPO over masked semantic candidates |
| First proof of value | Better accuracy at matched cost than a competent script and myopic acquisition |

The first useful milestone is a small repeatable experiment in which two investigations see the same incident, buy different evidence, and produce different accuracy-cost outcomes. Once that works, PPO has a clear problem to learn. Until then, increasing model size or training steps mostly makes it harder to tell which assumption failed.

A more defensible statement of the project would be:

> We study cost-sensitive sequential evidence acquisition for cloud root-cause analysis. A small policy selects parameterized diagnostic tools and a stopping decision from a structured representation of acquired evidence and resource dependencies. A separately evaluated evidence processor converts telemetry into diagnostic beliefs. We compare learned acquisition with scripted, myopic, and prompted alternatives under matched information and cost budgets, and measure transfer from simulation and replay to controlled cloud incidents.

That claim leaves room for the result to be positive, mixed, or negative. It also puts your RL work at the center of a testable implementation.
