# Graph-Based Intelligent Root Cause Analysis

AI-powered RCA system for cloud-native microservices. Combines a live Knowledge Graph (Neo4j) with GraphRAG-based dynamic causal expansion and an LLM chat interface to automatically detect, trace, and explain incidents.

---

## Architecture

```
Sock Shop (17 containers)
  └── Prometheus (metrics) + Loki (logs)
        └── kg_builder.py  — builds & polls live Neo4j Knowledge Graph
              └── retriever.py  — dynamic causal expansion (GraphRAG)
                    └── llm_chat.py  — LLM Q&A session (Gemini / Groq)
```

---

## Quick Start

### 1. Prerequisites
- Docker Desktop
- Python 3.10+
- Gemini or Groq API key

### 2. Setup

```bash
cd rca-project
bash setup.sh
source venv/bin/activate

cp pipeline/.env.example pipeline/.env
# Edit pipeline/.env and add your API key
```

### 3. Start the Stack

```bash
docker compose up -d
```

Wait ~60 seconds for all services to boot.

### 4. Run

```bash
# Terminal 1 — live monitor
python pipeline/monitor.py

# Terminal 2 — LLM chat
python pipeline/llm_chat.py

# Terminal 3 — inject faults
python pipeline/fault_injector.py
```

---

## Fault Scenarios

| # | Type | Target |
|:--|:-----|:-------|
| 1 | DB crash | catalogue-db |
| 2 | CPU burn | payment |
| 3 | Service crash | orders |
| 4 | Bad deploy | payment |
| 5 | Memory leak | carts |
| 6 | Upstream DB crash | orders-db |
| 7 | Auth failure | user |
| 8 | Multi-fault chaos | catalogue-db + payment |
| 9 | Silent downstream failure | shipping |

---

## Key URLs (after stack is up)

| Service | URL |
|:--------|:----|
| Sock Shop UI | http://localhost:80 |
| Prometheus | http://localhost:9090 |
| Neo4j Browser | http://localhost:7474 |
| Loki | http://localhost:3100 |

Neo4j login: `neo4j / password123`

---

## Pipeline Files

| File | Purpose |
|:-----|:--------|
| `kg_builder.py` | Builds Neo4j graph, polls Prometheus, detects anomalies |
| `retriever.py` | GraphRAG — dynamic causal expansion, fetches logs |
| `llm_chat.py` | LLM chat session with live anomaly watcher |
| `monitor.py` | Live health dashboard (replaces collect.py) |
| `fault_injector.py` | Interactive fault injection menu |
| `graphrag.py` | Thin standalone graph inspector |
| `verify_all.py` | Checks all services are reachable |
# Investigation integration on Windows

The new `rca_investigation` package uses Python 3.12 and a locked environment.
Run these commands in native PowerShell from the `team-graph-rca` directory:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --frozen --group dev
uv run --frozen rca-investigate validate-config --config configs/investigation_v1.yaml
uv run --frozen rca-investigate validate-fixtures
uv run --frozen pytest
uv run --frozen rca-investigate preflight --profile sockshop_v1
```

The default install requires no Torch, GPU, provider key or running lab.
Legacy LLM dependencies are optional, installed with `uv sync --extra legacy --extra explain`.
Keep this environment separate from WSL Python and the parent simulator environment.

The first task has three classes, `catalogue-db`, `orders` and `NO_FAULT`.
Its manual entry event always uses `front-end`. It exposes five status tools.
The seven synthetic fixtures verify acquisition, stopping, deferral, failure costs
and joint planning. Their results do not measure real Sock Shop accuracy.

`validate-config`, `validate-fixtures` and `preflight` write hashed reports to
`runs/integration_v1`. A failed check returns a nonzero exit code.
The live workload and both supported stop effects have passed physical checks.
The 30-incident capture pilot and separate 120-incident evaluation cohort are
complete. The existing chat and fault menu remain separate from this experiment.

```powershell
uv run --frozen rca-investigate pin-images
docker compose -p rca-lab -f docker-compose.yml -f configs/lab_pinned.yaml up -d
uv run --frozen rca-investigate verify-live
uv run --frozen rca-investigate capture --profile sockshop_v1 --schedule configs/capture_pilot.yaml
uv run --frozen rca-investigate capture --profile sockshop_v1 --schedule configs/capture_evaluation.yaml
uv run --frozen rca-investigate split --dataset datasets/sockshop_v1 --schedule-state runs/integration_v1/captures/sockshop_v1/schedule.json
uv run --frozen rca-investigate fit-model --split train
uv run --frozen rca-investigate train --policy expected-sarsa --episodes 10000 --seeds 11 22 33 44 55
uv run --frozen rca-investigate train --policy two-step-sarsa --episodes 10000 --seeds 11 22 33 44 55
uv run --frozen rca-investigate compare --split validation
uv run --frozen rca-investigate compare --split test --frozen-manifest experiments/local/final_v1.json
uv run --frozen rca-investigate investigate --backend replay --incident datasets/sockshop_v1/OPAQUE_ID --policy selected
```

Capture resumes accepted schedule slots. Failed effects are rejected; failed
recovery leaves `blocked-lab.json` and blocks further injection. Do not remove that
marker until the recorded target and workload have actually recovered. Hard
interruption can also leave `capture.lock`; inspect its PID before removing it.
The 30-incident mechanics pilot and 120-incident evaluation cohort have separate
directories. Session split roles are assigned before collection. Fitting reads
only training labels, and selection uses validation return with lower-cost ties.
Reserved test comparison requires a frozen source, task, split and checkpoint
manifest. A completed test comparison cannot be rerun through the CLI.

Training keeps `latest.json`, validation-selected `best.json` and metrics for each
seed. Resume requires `--resume` and unchanged task, inputs, source and settings.
`train-fixture` supports a bounded interruption check through `--stop-after`.
Expected SARSA and two-step SARSA share the same state, rewards and dataset.
Pure and planner-fallback tabular behavior are evaluated separately.

`pipeline/investigate.py` invokes the same public replay runner as the CLI. Its
explanation cites acquired evidence IDs. This is delayed investigation of a
frozen capture window. Ordinary Docker inspection cannot reconstruct past state.
After final selection, run `uv run --frozen python scripts/verify_selected_application.py`
to compare CLI/application decisions, actions and acquired evidence on one
validation capture per supported class.

Docker evidence uses the `rca-lab` Compose project. Status inspection refuses
historical windows. A capture controller must acquire and persist status while
the window is current. Prometheus process CPU and Docker container CPU are
different measurements. The first task keeps CPU and log actions disabled until
their data is measured and the task profile is explicitly promoted.

`monitor.py` and `collect.py` share `data/graph_writer.lock`; run one writer.
Startup updates topology without clearing live node state. An interrupted writer
may leave a lease. Inspect the recorded PID before removing a stale lease.

The initial storage failure was repaired after the user freed disk space. Some
cached image extractions were damaged. The experiment override pins repaired
images and uses verified Mongo 3.6.23 for carts-db and orders-db instead of the
damaged Mongo 3.4 image. The base legacy Compose file is retained. Image digests,
resource limits and live checks are recorded in `runs/integration_v1`.

AWS command preparation is described in `docs/aws_extension.md`. No AWS account,
deployment, executing backend or performance transfer is assumed.

## Completed stop_v1 experiment

The session splits contain 60 training, 30 validation and 30 reserved test
captures, balanced across the three classes. All 120 evaluation captures passed
physical effect and recovery checks. The pilot retains two rejected attempts.
Both tabular methods completed 10,000 replay episodes for each of five seeds.

Validation selected the dependency rule through the declared simplicity tie
break. The rule, planners and all selected tabular checkpoints scored 30/30 on
reserved test captures at 1.67 average credits and 0.9167 mean return. The two
test sessions are a small, correlated sample. RL and planning show no measured
gain over the rule on this task.

CLI/application decisions, semantic traces and acquired evidence match on one
validation capture per class. Final workload and telemetry checks pass.
The compact result and artifact checksums are in `experiments/stop_v1_result.json`.
Raw captures, checkpoints and full reports remain local in `datasets/` and `runs/`.

Synthetic status outage reduces rule accuracy to one third. The rule does not
defer on unsupported payment stops or simultaneous stops. Those challenge
results have separate reports and do not extend the supported diagnosis scope.
Additional physical challenges, payment contention, evolving live episodes and
an executing AWS backend need new measured experiments.

## Adaptive v2 implementation

The phase checklist and measured limits are in
`docs/adaptive_v2_implementation.md`. The registered profiles add payment CPU
contention (v2a), orders-to-payment delay (v2b), and retained carts memory pressure
(v2c). Memory pressure has not established customer degradation.

Raw observations are immutable and separate from paid policy packets. Frozen
session roles determine train/validation/test membership. Thresholds and observed
log-peer mappings are fitted from training only. An all-zero encoder hash means
unfitted; it is resolved from the profile's encoder artifact before replay.

```powershell
uv run --frozen rca-investigate validate-profile --config configs/investigation_v2a.yaml
uv run --frozen rca-investigate verify-profile --config configs/investigation_v2a.yaml
uv run --frozen rca-investigate validate-adaptive-fixtures
uv run --frozen python scripts/run_adaptive_experiment.py --config configs/investigation_v2a.yaml --schedule configs/capture_v2a_evaluation.yaml --reserved-test
```

`--wait-for-capture` lets the experiment runner follow an already running capture
controller. It fits and trains after collection completes and the validation
evidence gate passes. Captures resume accepted slots; interrupted or failed
recovery retains the lab block. A profile's complete 160/200/240-incident cohort
takes many hours because every incident uses the same 120/30/120-second protocol.

`scripts/run_adaptive_phases.py` runs v2a, v2b and v2c in order, including the
declared sweeps. Use `--first-capture-running` to follow an active v2a capture.
Between v2a and v2b it installs the pinned lease helper images, retains the
preceding reports, and repeats workload, inventory and fault verification.
It stops on a failed check and records the reason in
`runs/adaptive_v2/phase-sequence.json`.
On Windows, the runner prevents automatic system sleep while it is active.
The request ends when the process exits; display power settings stay in effect.

Use `declare-ablations` and `declare-cost-sweep` with `--config` and a new directory
in `--output` to create separate experiments over the same raw cohort. Each
variant keeps all diagnoses and needs its own model and trained checkpoints.
The experiment runner's `--sweeps` option declares the variants before opening
reserved comparisons, then runs them after the main profile. Weak subsets keep
all roots and get trained checkpoints; their failed evidence gate stays in the
report and prevents a promotion claim.

Engineering pilots and controlled decision fixtures do not establish physical
policy performance. Reserved comparisons freeze policies and source identities
before test evaluation. Pure learners and hybrids have separate reports;
source-outage perturbations remain synthetic.
