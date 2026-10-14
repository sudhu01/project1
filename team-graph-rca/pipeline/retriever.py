"""
retriever.py — Pure GraphRAG Retrieval Engine
Responsibility: traverse the Neo4j KG, extract logs, return a structured context dict.
No LLM, no printing, no prompt building. Just clean graph retrieval.
"""

import subprocess
import requests
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# LangChain Neo4jGraph (with native fallback — zero APOC)
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    try:
        from langchain_community.graphs import Neo4jGraph
    except ImportError:
        Neo4jGraph = None

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password123"
LOKI_URL   = "http://localhost:3100"
MAX_HOPS   = 5


import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)

def _make_graph():
    """Returns a Neo4j query-capable object (LangChain or native fallback)."""
    if Neo4jGraph is not None:
        try:
            return Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASS,
                enhanced_schema=False,
                refresh_schema=False,
            )
        except Exception:
            pass
    # Pure native driver fallback — identical .query() interface with notifications disabled
    from neo4j import GraphDatabase, NotificationMinimumSeverity
    class _Native:
        def __init__(self):
            try:
                self._d = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASS),
                    notifications_min_severity=NotificationMinimumSeverity.OFF
                )
            except Exception:
                self._d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

        def query(self, q, params=None):
            with self._d.session() as s:
                return [dict(r) for r in s.run(q, params or {})]
        def close(self):
            self._d.close()
    return _Native()


def get_active_anomalies(graph):
    """Return all services currently flagged anomalous in Neo4j, ordered by earliest anomaly."""
    return graph.query("""
        MATCH (n)
        WHERE n.anomaly_flag = true
        RETURN n.name        AS name,
               n.role        AS role,
               n.criticality AS criticality,
               n.anomaly_type AS anomaly_type,
               n.anomaly_since AS anomaly_since,
               n.cpu_usage   AS cpu_usage,
               n.memory_usage AS memory_usage,
               n.error_rate  AS error_rate,
               n.status      AS status
        ORDER BY n.anomaly_since ASC
    """)


def dynamic_expansion(graph, start_service):
    """
    Hop-by-hop expansion starting from `start_service`.
    - Expands along :CALLS and :TRIGGERS edges.
    - Continues into anomalous neighbors.
    - Stops at healthy neighbors (records them as healthy perimeter).
    - Capped at MAX_HOPS. Returns rare-case analysis if cap is hit.
    """
    visited   = {start_service}
    frontier  = {start_service}
    anomalous = {}
    healthy   = {}
    edges     = []
    hop_trace = []

    # Bootstrap: fetch the starting node
    res = graph.query("""
        MATCH (n {name: $name})
        RETURN n.name AS name, n.role AS role, n.category AS category,
               n.criticality AS criticality, n.status AS status,
               n.anomaly_flag AS anomaly_flag, n.anomaly_type AS anomaly_type,
               n.anomaly_since AS anomaly_since, n.cpu_usage AS cpu_usage,
               n.memory_usage AS memory_usage, n.error_rate AS error_rate,
               n.what_it_calls AS what_it_calls, n.who_calls_it AS who_calls_it
    """, params={"name": start_service})

    if res:
        node = res[0]
        if node.get("anomaly_flag"):
            anomalous[start_service] = node
        else:
            healthy[start_service] = node

    hops_done = 0
    for hop in range(1, MAX_HOPS + 1):
        if not frontier:
            break
        hops_done = hop
        next_frontier  = set()
        new_anomalies  = []

        for current in frontier:
            neighbors = graph.query("""
                MATCH (curr {name: $name})-[r:CALLS|TRIGGERS]-(nbr)
                RETURN curr.name                AS curr_name,
                       type(r)                  AS rel_type,
                       startNode(r).name        AS edge_src,
                       endNode(r).name          AS edge_tgt,
                       nbr.name                 AS name,
                       nbr.role                 AS role,
                       nbr.category             AS category,
                       nbr.criticality          AS criticality,
                       nbr.status               AS status,
                       nbr.anomaly_flag         AS anomaly_flag,
                       nbr.anomaly_type         AS anomaly_type,
                       nbr.anomaly_since        AS anomaly_since,
                       nbr.cpu_usage            AS cpu_usage,
                       nbr.memory_usage         AS memory_usage,
                       nbr.error_rate           AS error_rate,
                       nbr.what_it_calls        AS what_it_calls,
                       nbr.who_calls_it         AS who_calls_it
            """, params={"name": current})

            for rec in neighbors:
                edge = {
                    "source": rec["edge_src"],
                    "target": rec["edge_tgt"],
                    "type": rec["rel_type"],
                }
                if edge not in edges:
                    edges.append(edge)

                nbr_name = rec["name"]
                if nbr_name in visited:
                    continue
                visited.add(nbr_name)

                if rec.get("anomaly_flag"):
                    anomalous[nbr_name] = rec
                    next_frontier.add(nbr_name)
                    new_anomalies.append(nbr_name)
                else:
                    healthy[nbr_name] = rec   # stop expanding this branch

        hop_trace.append({
            "hop": hop,
            "from": list(frontier),
            "new_anomalies_found": new_anomalies,
        })
        frontier = next_frontier

    capped = bool(frontier)  # still unexplored anomalous nodes at cap

    # Root cause = anomalous node with earliest timestamp (or leaf with no callers)
    candidate_root = start_service
    if anomalous:
        candidate_root = sorted(
            anomalous.values(),
            key=lambda x: x.get("anomaly_since") or "9999"
        )[0]["name"]

    rare_case = None
    if capped:
        rare_case = {
            "flag": "RARE_CASE_CASCADE_OVERFLOW",
            "message": (
                f"Expansion hit the {MAX_HOPS}-hop cap without reaching a healthy boundary. "
                "Likely a circular dependency or widespread cascading collapse."
            ),
            "uncontained_nodes": list(frontier),
            "structural_hints": {
                n: {
                    "calls": anomalous.get(n, {}).get("what_it_calls", []),
                    "called_by": anomalous.get(n, {}).get("who_calls_it", []),
                }
                for n in frontier
            },
        }

    return {
        "start_service":    start_service,
        "candidate_root":   candidate_root,
        "hops_done":        hops_done,
        "capped":           capped,
        "rare_case":        rare_case,
        "anomalous_nodes":  list(anomalous.values()),
        "healthy_perimeter": list(healthy.values()),
        "traversal_edges":  edges,
        "hop_trace":        hop_trace,
    }


def fetch_logs(service_name, limit=6):
    """Loki query with case-insensitive error pattern + Docker container log fallback."""
    query = f'{{container=~".*{service_name}.*"}} |~ "(?i)(error|exception|fail|fatal|panic|refused|crash)"'
    try:
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={"query": query, "limit": limit},
            timeout=2,
        )
        if resp.status_code == 200:
            logs = []
            for stream in resp.json().get("data", {}).get("result", []):
                for _, line in stream.get("values", []):
                    line = line.strip()
                    if line and line not in logs:
                        logs.append(line)
            if logs:
                return logs[:limit]
    except Exception:
        pass

    # Docker fallback
    try:
        cmd = f"docker logs --tail 20 $(docker ps -aq --filter name={service_name} | head -n 1) 2>&1"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
        raw = (res.stdout + res.stderr).splitlines()
        cues = [l.strip() for l in raw if any(
            k in l.lower() for k in ["err","fail","fatal","warn","exit","panic","exception","refus","crash"]
        )]
        return cues[-limit:] if cues else [l.strip() for l in raw[-limit:] if l.strip()]
    except Exception:
        pass

    return []


def build_context(graph, alerted_service):
    """
    Master retrieval call.
    Returns a structured context dict with everything the LLM needs:
    - Traversal path
    - Anomalous node features
    - Healthy perimeter
    - Extracted logs for every anomalous service
    - Rare case diagnostic (if cap hit)
    """
    expansion = dynamic_expansion(graph, alerted_service)

    # Gather logs for all anomalous services
    logs = {}
    for svc in set([alerted_service, expansion["candidate_root"]] +
                   [n["name"] for n in expansion["anomalous_nodes"]]):
        svc_logs = fetch_logs(svc, limit=4)
        if svc_logs:
            logs[svc] = svc_logs

    propagation_chain = [
        f"{e['source']} -> {e['target']}"
        for e in expansion["traversal_edges"]
        if e["type"] == "TRIGGERS"
    ]

    return {
        "incident": {
            "alerted_service":          alerted_service,
            "candidate_root_cause":     expansion["candidate_root"],
            "anomaly_type":             expansion["anomalous_nodes"][0].get("anomaly_type") if expansion["anomalous_nodes"] else "unknown",
            "hops_traversed":           expansion["hops_done"],
            "stopped_at_healthy_boundary": not expansion["capped"],
            "timestamp_utc":            datetime.now(timezone.utc).isoformat(),
        },
        "traversal": {
            "hop_trace":            expansion["hop_trace"],
            "propagation_chain":    propagation_chain,
            "all_edges":            expansion["traversal_edges"],
        },
        "anomalous_nodes":  expansion["anomalous_nodes"],
        "healthy_perimeter": [
            {"name": h["name"], "role": h["role"], "status": "healthy"}
            for h in expansion["healthy_perimeter"]
        ],
        "logs":             logs,
        "rare_case":        expansion["rare_case"],
    }
