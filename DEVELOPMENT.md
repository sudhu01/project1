# Local development environment

Steps 3 through 6.2 of the small simulator execution plan were completed on 2026-09-14.

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

`rca_sim.belief.ExactBeliefEstimator` maintains the normalized float64
posterior over every service, fault, and workload hypothesis. It updates from
new metric and log records using the generator's public likelihood functions,
deduplicates repeated evidence IDs, and exposes immutable service/fault,
service, and busy-workload marginals. Diagnosis ties use the lowest service ID
and then the lowest fault encoding.
