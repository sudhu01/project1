# Graph RCA — Demo Reference

## Live URLs

| Service | URL | Login |
|:--------|:----|:------|
| Sock Shop UI | http://localhost:80 | — |
| Prometheus | http://localhost:9090 | — |
| Neo4j Browser | http://localhost:7474 | `neo4j / password123` |
| Loki (API) | http://localhost:3100/ready | — |

---

## Run Order

```bash
# 1. Start the stack
cd /Users/Hakim/Desktop/projects/Graph_rca/rca-project
docker compose up -d

# 2. Activate venv (every new terminal)
source venv/bin/activate

# 3. Terminal 1 — Live monitor
python pipeline/monitor.py

# 4. Terminal 2 — LLM Chat
python pipeline/llm_chat.py

# 5. Terminal 3 — Inject faults
python pipeline/fault_injector.py
```

---

## Neo4j Cypher Queries

```cypher
-- Full topology graph
MATCH (n)-[r]->(m) RETURN n, r, m

-- Only anomalous nodes
MATCH (n) WHERE n.anomaly_flag = true RETURN n

-- Active fault propagation edges
MATCH (a)-[r:TRIGGERS]->(b) RETURN a, r, b

-- All node metrics
MATCH (n) WHERE n.role IS NOT NULL
RETURN n.name, n.status, n.cpu_usage, n.memory_usage, n.error_rate
ORDER BY n.name
```

---

## Prometheus Queries

```promql
# All scraped services (should all be 1 = up)
up

# CPU across services
rate(process_cpu_seconds_total[1m]) * 100

# Memory (MB)
process_resident_memory_bytes / 1000000

# JVM heap (Spring Boot services)
jvm_memory_used_bytes{area="heap"} / 1000000
```

---

## Fault Scenarios

| # | Type | Target | Description |
|:--|:-----|:-------|:------------|
| 1 | dependency_failure | catalogue-db | DB crash → catalogue goes down |
| 2 | resource_exhaustion | payment | CPU burn → payment degraded |
| 3 | cascading_failure | orders | Orders crash → cascades to frontend |
| 4 | config_deploy | payment | Bad deploy simulation |
| 5 | resource_exhaustion | carts | Memory leak → slow OOM |
| 6 | upstream_dependency | orders-db | DB crash → orders times out silently |
| 7 | auth_failure | user | Auth service down → login broken everywhere |
| 8 | cascading_failure | catalogue-db + payment | Multi-fault chaos |
| 9 | downstream_failure | shipping | Shipping down → orders complete but fulfilment breaks |

---

## Neo4j Credentials

```
URI:      bolt://localhost:7687
User:     neo4j
Password: password123
```
