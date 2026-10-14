# Fault coverage and manual agent testing

14 October 2026

The completed adaptive benchmark covers five fault scenarios and a healthy class. They are catalogue-db stopped, orders stopped, payment CPU contention, orders-to-payment delay and retained carts memory pressure. There are four fault mechanisms: stopping a container, CPU contention, path delay and memory pressure. The policy diagnoses registered causes; it does not separately classify every possible mechanism at a resource.

The upstream graph-rca injector has nine named menu scenarios. Three match our measured scenarios closely: catalogue-db stop, orders stop and payment CPU burn. Its carts memory entry overlaps the evidence family, but our benchmark establishes retained pressure, not a demonstrated leak. Payment stop under the name bad deployment, orders-db stop, user stop, simultaneous catalogue-db/payment faults and shipping stop have not been trained or benchmarked in the adaptive agents. Our measured payment-path delay is an addition. The upstream source is https://github.com/the-jellyone/graph-rca/blob/main/pipeline/fault_injector.py.

## How the current agents run

The original monitor/chat/fault-injector terminals run the GraphRAG demo. They do not invoke the trained VOI, DP or SARSA agents. The benchmark agents currently replay saved capture windows. There is no supported continuously running live agent command. Newly injected faults require a separate fresh capture workflow before these agents can investigate them. Do not send a new capture into the accepted benchmark directories.

Use three native PowerShell terminals on the Windows host, not shells inside the Sock Shop containers. Set this working directory in each:

```powershell
Set-Location 'C:\Users\jacks\OneDrive\Desktop\Project1\team-graph-rca'
```

## Terminal 1: existing Docker lab

Start Docker Desktop with Linux containers. Then:

```powershell
docker info
$labIds = @(docker ps -aq --filter 'label=com.docker.compose.project=rca-lab')
if ($labIds.Count -eq 0) { throw 'The saved rca-lab containers are missing.' }
docker start $labIds
docker ps --filter 'label=com.docker.compose.project=rca-lab'
$frontId = @(docker ps -aq --filter 'label=com.docker.compose.project=rca-lab' --filter 'label=com.docker.compose.service=front-end')
if ($frontId.Count -ne 1) { throw 'Expected one registered front-end container.' }
docker logs --follow --tail 30 $frontId[0]
```

This starts existing containers and follows frontend logs. Ctrl+C stops following the logs. It does not stop Docker containers. Replay itself does not require Docker, so this terminal is optional when only comparing saved incidents. Do not recreate the lab with the base Compose file; the measured lab includes pinned images and overrides.

## Terminal 2: choose a saved incident

```powershell
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --list
```

Available case names are `catalogue-db`, `orders`, `payment-cpu`, `payment-delay`, `carts-memory` and `healthy`. Each has ten validation incidents. Use `--sample 1` through `--sample 10` to choose one. These labels select a demonstration case and score the completed result; they are not supplied to the policy's observation.

## Terminal 3: run the agents

```powershell
# All four agents on the same payment CPU case.
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-cpu --sample 1 --policy all

# Run them separately.
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-cpu --policy voi
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-cpu --policy dp
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-cpu --policy expected-sarsa --seed 11
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-cpu --policy two-step-sarsa --seed 11

# Change the case or learner seed.
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case carts-memory --sample 2 --policy all
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-delay --policy expected-sarsa --seed 22

# Explicit planner-fallback variant.
.\.venv\Scripts\python.exe scripts/demo_saved_incident.py --case payment-delay --policy expected-sarsa --seed 22 --hybrid
```

Without `--hybrid`, SARSA uses the learned table alone. Available seeds are 11, 22, 33, 44 and 55. VOI and DP do not use learner seeds. The output shows diagnosis, confidence, correctness, probes, credits, score and the action trace with computation time. Detailed JSON goes into `runs/manual_replay`, separate from frozen benchmark reports. No training or physical injection occurs.

The script's payment CPU sample was verified with all four agents. VOI and DP each diagnosed payment correctly using two probes and score 0.8. Seed-11 Expected SARSA used five probes and score 0.65; two-step SARSA used four and score 0.7. This is one demonstration incident, not a replacement for the full benchmark.

## Original GraphRAG three-terminal demo

For the separate original application, its documented commands are:

```powershell
# Terminal 1
.\.venv\Scripts\python.exe pipeline/monitor.py

# Terminal 2
.\.venv\Scripts\python.exe pipeline/llm_chat.py

# Terminal 3
.\.venv\Scripts\python.exe pipeline/fault_injector.py
```

That application needs its optional LLM dependencies and configured provider key. The legacy menu has different fault implementations and broader targets than the measured adaptive benchmark. Recover the current fault with `r` before selecting another. These commands are not the agent-testing workflow above, and I did not run any live injections while preparing this guide.
