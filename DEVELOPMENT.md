# Local development environment

Steps 3 through 8.4 of the small simulator execution plan were completed on 2026-09-14.

- Python: CPython 3.12.13, 64-bit
- uv: 0.11.30
- PyTorch: 2.14.0+cpu
- Gymnasium: 1.3.0

`uv.lock` records the complete resolved dependency set. Use locked commands after the initial setup:

```powershell
uv sync --locked
uv run --locked pytest -q
```

`rca_sim.graph` contains the dependency DAG generator and graph queries. Edges
point from callers to dependencies. Generated graphs use randomized public
service IDs even though construction starts in a temporary topological order.

`rca_sim.world` defines and samples the private incident hypothesis. Cause
service, fault type, and workload use independent uniform priors. The entry
alert does not filter or otherwise change that prior.

`classify_service_relationship` is the single relationship rule for world
generation and public likelihood calculations. It follows edges from an
observed caller toward the candidate failed dependency and reports the shortest
directed distance.

`rca_sim.likelihoods` owns the metric and log probability model used by both
world generation and exact inference. A generated observation set contains two
binary metric readings per category and three categorical log readings per
service. These arrays are copied and made read-only after sampling.

`generate_incident_world` freezes the graph, hidden hypothesis, observations,
and 72 evidence records for the default eight-service incident. Evidence IDs
derive from validated record fields, reads are immutable lookups, and the world
retains the seed and generator settings required for replay.

`rca_sim.tools` defines the four semantic probe templates in stable index order.
Each template owns its fixed credit cost and evidence coverage. Probe collection
reads the world's frozen records, works for every active service, and never
resamples or mutates the incident.

The action mapping assigns four probe indices to each of ten padded service
slots. Probe indices are 0 through 39, and STOP is always index 40 in a
41-action discrete space. Encoding and decoding preserve the semantic template
instead of treating each slot as an unrelated action.

The canonical action mask rejects padded services, unavailable tools,
unaffordable probes, and probes whose complete coverage is already in the
evidence ledger. STOP remains valid until termination. The simulator executor
validates against this mask before reading the frozen world and returns an
immutable, backend-neutral result with semantic action, evidence, cost, and
coverage fields.

`rca_sim.evidence.EvidenceLedger` deduplicates exact repeats by evidence ID.
It rejects a conflicting value atomically, before adding any record from the
batch.

`rca_sim.belief.ExactBeliefEstimator` maintains unnormalized float64 log
probabilities over every service, fault, and workload hypothesis. It updates
from new metric and log records using the generator's public likelihood
functions, deduplicates repeated evidence IDs, and normalizes through a stable
log-sum-exp calculation. Zero likelihoods become negative infinity, while
evidence that rules out every hypothesis raises `InconsistentEvidenceError`.
The separate reference path recomputes from the uniform prior and the complete
deduplicated ledger. The estimator exposes immutable service/fault, service,
and busy-workload marginals. Diagnosis ties use the lowest service ID and then
the lowest fault encoding.

The hand-checkable two-hypothesis test starts from equal prior mass. Evidence
with likelihoods 0.8 and 0.2 produces that same posterior split. A second
record with likelihoods 0.25 and 0.75 changes the split to 4/7 and 3/7.
Reading either record again leaves the posterior unchanged.

`rca_sim.observation` declares the versioned `sim_v0` Gymnasium dictionary
space and its fixed 42 node and 10 global feature columns. The builder uses
only acquired evidence, inferred beliefs, public graph data, resource state,
and executed probe history. It returns fresh padded arrays and rejects values
outside the declared schema bounds instead of clipping them.

`rca_sim.environment.InvestigationEnv` implements reproducible Gymnasium
reset and probe transitions. Reset creates an independent frozen incident or
loads an exact case descriptor, then clears evidence, belief, budget, and
history state. Probe steps save the prior observation, charge the template
cost, update belief from unseen evidence, and auto-finalize on budget, horizon,
or lack of another feasible probe. STOP ends immediately at zero cost and
scores the current service diagnosis. Every episode return is terminal
correctness minus `lambda_cost` times total credits spent.

Normal environment steps always return `truncated=False`. STOP and public
resource limits are true terminations, and the environment rejects another
step until reset so an incident cannot be scored twice. Collectors can copy
the live observation at a batch cutoff without changing episode state, then
bootstrap from it and continue the same incident in the next batch. External
Gymnasium wrappers may still report their own truncation separately.

`python -m rca_sim.validate` runs the step 8.1 and 8.2 checks with fixed seeds.
It checks every probability table, distance decay, 10,000 observation samples
for each selected fixed hypothesis, and independent prior sampling. It also
checks posterior normalization, reference agreement, evidence order, scope
upgrades, duplicate reads, and shared-workload marginalization. Pass `--output`
to save the JSON report used as the validation artifact.

`rca_sim.fixtures` defines nine finite decision fixtures. Each fixture lists its
valid latent hypotheses, prior, binary evidence likelihoods, enabled actions,
coverage, costs, and optional public initial evidence. `FixtureEnv` uses the
same padded `sim_v0` observation and 41-action schemas as the main environment.
Its exact finite-state planner reports one-step or multi-step action values and
favors STOP on ties. The step 8.3 validation checks all required decisions,
including the two-bit complementarity case where depth one stops and depth two
investigates.

`InvestigationEnv(trace_enabled=True)` records public reset, probe, and STOP
events without reading the environment RNG or exposing the hidden incident
label. The step 8.4 validator samples 10,000 valid transitions and checks masks,
budgets, finite observations, padded actions, terminal payout protection, the
last affordable probe, invalid-action atomicity, private-label isolation, and
trace equivalence. The transition sample count cannot be set below 10,000.
