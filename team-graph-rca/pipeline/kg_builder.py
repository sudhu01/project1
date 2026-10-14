"""
Knowledge Graph Builder for Cloud-Native Microservices (Sock Shop)
Builds topological architecture, ingests real-time Prometheus telemetry,
and dynamically tracks anomaly propagation in Neo4j using the LangChain Neo4jGraph wrapper.
"""

import time
import requests
from datetime import datetime, timezone

# LangChain Neo4jGraph Connection Layer
try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    try:
        from langchain_neo4j import Neo4jGraph
    except ImportError:
        from neo4j import GraphDatabase
        class Neo4jGraph:
            def __init__(self, url, username, password):
                self.driver = GraphDatabase.driver(url, auth=(username, password))
            def query(self, query, params=None):
                with self.driver.session() as session:
                    result = session.run(query, params or {})
                    return [dict(r) for r in result]
            def close(self):
                if hasattr(self, 'driver') and self.driver:
                    self.driver.close()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password123"
PROMETHEUS = "http://localhost:9090"
POLL_INTERVAL = 30  # seconds

SERVICES_METADATA = {
    "front-end": {
        "role": "API Gateway & Customer Web UI",
        "category": "Service",
        "criticality": "critical",
        "calls": ["catalogue", "carts", "orders", "user"],
    },
    "catalogue": {
        "role": "Product Catalogue Service",
        "category": "Service",
        "criticality": "high",
        "calls": ["catalogue-db"],
    },
    "catalogue-db": {
        "role": "Product MySQL Database",
        "category": "Database",
        "criticality": "critical",
        "calls": [],
    },
    "carts": {
        "role": "Cart Management Service",
        "category": "Service",
        "criticality": "high",
        "calls": ["carts-db"],
    },
    "carts-db": {
        "role": "Cart Mongo Database",
        "category": "Database",
        "criticality": "high",
        "calls": [],
    },
    "orders": {
        "role": "Order Processing Service",
        "category": "Service",
        "criticality": "critical",
        "calls": ["payment", "shipping", "orders-db"],
    },
    "orders-db": {
        "role": "Order Mongo Database",
        "category": "Database",
        "criticality": "critical",
        "calls": [],
    },
    "payment": {
        "role": "Payment Authorization Service",
        "category": "Service",
        "criticality": "critical",
        "calls": [],
    },
    "user": {
        "role": "User Account & Authentication Service",
        "category": "Service",
        "criticality": "medium",
        "calls": ["user-db"],
    },
    "user-db": {
        "role": "User Profile Mongo Database",
        "category": "Database",
        "criticality": "medium",
        "calls": [],
    },
    "shipping": {
        "role": "Dispatch & Shipping Service",
        "category": "Service",
        "criticality": "medium",
        "calls": ["queue-master"],
    },
    "queue-master": {
        "role": "Async Queue Worker Service",
        "category": "Service",
        "criticality": "low",
        "calls": ["rabbitmq"],
    },
    "rabbitmq": {
        "role": "AMQP Message Broker",
        "category": "Queue",
        "criticality": "medium",
        "calls": [],
    },
}

for svc, meta in SERVICES_METADATA.items():
    callers = [
        caller for caller, c_meta in SERVICES_METADATA.items()
        if svc in c_meta["calls"]
    ]
    meta["who_calls_it"] = callers


class KGBuilder:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS, prometheus_url=PROMETHEUS):
        self.uri = uri
        self.user = user
        self.password = password
        self.prometheus_url = prometheus_url
        try:
            # Tell LangChain NOT to query apoc.meta.data()
            self.graph = Neo4jGraph(
                url=self.uri,
                username=self.user,
                password=self.password,
                enhanced_schema=False,
                refresh_schema=False
            )
        except Exception:
            # Pure native fallback wrapper with exact same .query() interface (zero APOC)
            from neo4j import GraphDatabase
            class _PureNeo4j:
                def __init__(self, u, usr, pwd):
                    self._d = GraphDatabase.driver(u, auth=(usr, pwd))
                def query(self, q, params=None):
                    with self._d.session() as s:
                        res = s.run(q, params or {})
                        return [dict(r) for r in res]
                def close(self):
                    self._d.close()
            self.graph = _PureNeo4j(self.uri, self.user, self.password)

        print(f"Connected to Neo4j via LangChain Neo4jGraph (APOC disabled) at {self.uri}")

    def close(self):
        if hasattr(self.graph, "close"):
            self.graph.close()
            print("Neo4j connection closed.")

    def clear_graph(self):
        self.graph.query("MATCH (n {lab_project: $project}) WHERE n.name IN $names DETACH DELETE n",
                         params={"project": "rca-lab", "names": list(SERVICES_METADATA)})
        print("Graph database cleared.")

    def create_service_nodes(self):
        for name, meta in SERVICES_METADATA.items():
            label = meta["category"]
            self.graph.query(f"""
                MERGE (n:{label} {{name: $name}})
                ON CREATE SET n.status = 'unknown',
                    n.anomaly_flag = false,
                    n.anomaly_type = '',
                    n.anomaly_since = null,
                    n.cpu_usage = null,
                    n.memory_usage = null,
                    n.error_rate = null,
                    n.latency_p99 = null,
                    n.last_updated = $ts
                SET n.role = $role,
                    n.category = $category,
                    n.criticality = $criticality,
                    n.what_it_calls = $calls,
                    n.who_calls_it = $who_calls_it,
                    n.lab_project = 'rca-lab'
            """,
            params={
                "name": name,
                "role": meta["role"],
                "category": meta["category"],
                "criticality": meta["criticality"],
                "calls": meta["calls"],
                "who_calls_it": meta["who_calls_it"],
                "ts": datetime.now(timezone.utc).isoformat()
            })
        print(f"Created {len(SERVICES_METADATA)} service nodes with complete metadata.")

    def create_topology_edges(self):
        total_edges = 0
        for source, meta in SERVICES_METADATA.items():
            for target in meta["calls"]:
                self.graph.query("""
                    MATCH (a {name: $source})
                    MATCH (b {name: $target})
                    MERGE (a)-[r:CALLS]->(b)
                    ON CREATE SET r.avg_latency = 0.0,
                        r.error_rate = 0.0
                """, params={"source": source, "target": target})
                total_edges += 1
        print(f"Created {total_edges} architectural [:CALLS] edges.")

    def query_prometheus(self, query):
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") != "success" or payload.get("warnings"):
                    return []
                data = payload.get("data", {})
                return data.get("result", [])
        except Exception:
            pass
        return []

    def is_container_running(self, service_name):
        """Return True, False, or None from exact Compose identity and state."""
        if service_name not in SERVICES_METADATA:
            raise ValueError("unknown service")
        try:
            import subprocess
            from rca_investigation.backends.docker import compose_rows, resolve_containers
            result = subprocess.run(["docker", "compose", "-p", "rca-lab", "ps", "--all", "--no-trunc", "--format", "json"],
                                    capture_output=True, text=True, timeout=5, shell=False)
            if result.returncode != 0:
                return None
            rows = compose_rows(result.stdout)
            ids = resolve_containers(rows, [service_name])
            state = next(row["State"] for row in rows if row["ID"] == ids[service_name])
            return True if state == "running" else False if state in ("exited", "dead") else None
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            return None

    def _first_metric(self, query, metric, units):
        import math
        result = self.query_prometheus(query)
        if len(result) != 1:
            return None
        try:
            value = float(result[0]["value"][1])
            sample = float(result[0]["value"][0])
            if not math.isfinite(value) or value < 0 or not math.isfinite(sample):
                return None
        except (ValueError, KeyError, TypeError):
            return None
        self._metric_receipts[metric] = {"availability": "OK", "source": "prometheus", "sample_time": sample, "units": units}
        return value

    def _instance(self, service_name):
        if service_name not in SERVICES_METADATA:
            raise ValueError("unknown service")
        return f"{service_name}:8079" if service_name == "front-end" else f"{service_name}:80"

    def _query_cpu(self, service_name):
        instance = self._instance(service_name)
        for query in [f'rate(process_cpu_seconds_total{{job="sockshop",instance="{instance}"}}[1m]) * 100',
                      f'process_cpu_usage{{job="sockshop",instance="{instance}"}} * 100']:
            value = self._first_metric(query, "cpu", "application_process_cpu_percent")
            if value is not None:
                return value
        return None

    def _query_mem(self, service_name):
        instance = self._instance(service_name)
        for query in [f'process_resident_memory_bytes{{job="sockshop",instance="{instance}"}}',
                      f'jvm_memory_used_bytes{{job="sockshop",instance="{instance}",area="heap"}}']:
            value = self._first_metric(query, "memory", "MB")
            if value is not None:
                return value / 1e6
        return None

    def _query_err(self, service_name):
        instance = self._instance(service_name)
        for query in [f'rate(http_requests_total{{job="sockshop",instance="{instance}",status=~"5.."}}[1m])',
                      f'rate(http_server_requests_seconds_count{{job="sockshop",instance="{instance}",status=~"5.."}}[1m])']:
            value = self._first_metric(query, "errors", "5xx_events_per_second")
            if value is not None:
                return value
        return None

    def update_node_metrics(self, service_name, grace=False):
        """Update Neo4j node with live metrics. grace=True skips anomaly detection."""

        self._metric_receipts = {}
        container_running = self.is_container_running(service_name)
        cpu = self._query_cpu(service_name)
        mem = self._query_mem(service_name)
        err = self._query_err(service_name)

        up_res = self.query_prometheus(f'up{{job="sockshop",instance="{self._instance(service_name)}"}}')
        prom_up = float(up_res[0]["value"][1]) == 1.0 if len(up_res) == 1 else None
        # Failed scrapes are telemetry failures, not proof of stopped containers.
        is_crash = container_running is False
        unknown = container_running is None or prom_up is False or cpu is None or mem is None or err is None

        anomaly      = False
        anomaly_type = None

        if not grace:
            if is_crash:
                anomaly      = True
                anomaly_type = "service_crash"
            elif cpu is not None and cpu > 85.0:
                anomaly      = True
                anomaly_type = "cpu_exhaustion"
            elif err is not None and err > 0.05:
                anomaly      = True
                anomaly_type = "high_error_rate"
            elif mem is not None and mem > 1000.0:
                anomaly      = True
                anomaly_type = "memory_exhaustion"

        ts = datetime.now(timezone.utc).isoformat()
        self.graph.query("""
            MATCH (n {name: $name})
            SET n.cpu_usage    = $cpu,
                n.memory_usage = $mem,
                n.error_rate   = $err,
                n.anomaly_flag = $anomaly,
                n.anomaly_type = CASE WHEN $anomaly THEN $anomaly_type ELSE null END,
                n.anomaly_since = CASE
                    WHEN $anomaly AND n.anomaly_since IS NULL THEN $ts
                    WHEN $anomaly THEN n.anomaly_since
                    ELSE null END,
                n.status       = CASE WHEN $anomaly THEN 'anomalous' WHEN $unknown THEN 'unknown' ELSE 'healthy' END,
                n.telemetry_availability = $availability,
                n.telemetry_source = 'prometheus',
                n.telemetry_receipts = $receipts,
                n.container_running = $container_running,
                n.metrics_scrape_up = $prom_up,
                n.error_rate_units = '5xx_events_per_second',
                n.last_updated = $ts
        """, params={
            "name":         service_name,
            "cpu":          None if cpu is None else round(cpu, 2),
            "mem":          None if mem is None else round(mem, 2),
            "err":          None if err is None else round(err, 4),
            "unknown": unknown,
            "availability": "NO_DATA" if unknown else "OK",
            "container_running": container_running,
            "prom_up": prom_up,
            "receipts": __import__("json").dumps(self._metric_receipts, sort_keys=True),
            "anomaly":      anomaly,
            "anomaly_type": anomaly_type,
            "ts":           ts,
        })

        if not anomaly:
            self.graph.query(
                "MATCH (n {name: $name})-[r:TRIGGERS]->() DELETE r",
                params={"name": service_name}
            )

        if anomaly:
            print(f"⚠️  ANOMALY: [{service_name}] -> {anomaly_type} "
                  f"(CPU: {cpu}, ERR events/s: {err}, MEM MB: {mem})")
        return anomaly

    def create_triggers_edge(self, source, target, fault_type="propagation"):
        """Creates a directed TRIGGERS edge representing fault propagation from source to target."""
        self.graph.query("""
            MATCH (a {name: $source})
            MATCH (b {name: $target})
            MERGE (a)-[r:TRIGGERS]->(b)
            SET r.fault_type = $fault_type,
                r.propagation_time = $ts,
                r.confidence = null,
                r.evidence_kind = 'rule_propagation_hypothesis'
        """, params={"source": source, "target": target, "fault_type": fault_type, "ts": datetime.now(timezone.utc).isoformat()})
        print(f"  🔗 PROPAGATION: ({source}) -[:TRIGGERS]-> ({target})")

    def poll_and_propagate(self, grace=False):
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Polling {len(SERVICES_METADATA)} services"
              + (" (grace period — no anomaly detection)" if grace else "") + "...")
        anomalous = []

        for svc in SERVICES_METADATA:
            if self.update_node_metrics(svc, grace=grace):
                anomalous.append(svc)

        if anomalous:
            for svc in anomalous:
                for caller in SERVICES_METADATA[svc].get("who_calls_it", []):
                    self.create_triggers_edge(svc, caller)
            print(f"🔴 Total anomalous services: {anomalous}")
        else:
            self.graph.query("MATCH (a {lab_project: 'rca-lab'})-[r:TRIGGERS]->(b {lab_project: 'rca-lab'}) DELETE r")
            if not grace:
                print("No confirmed resource anomaly. Missing telemetry may remain.")

        return anomalous

    def build_initial_graph(self, reset=False):
        print("=== Initializing Sock Shop Knowledge Graph ===")
        if reset:
            self.clear_graph()
        self.create_service_nodes()
        self.create_topology_edges()
        print("Knowledge Graph Successfully Initialized in Neo4j.")

    # Aliases for compatibility
    def build(self):
        self.build_initial_graph()

    def poll_all_services(self):
        return self.poll_and_propagate()

    def run_polling_loop(self, interval=POLL_INTERVAL):
        print(f"Starting live metric polling loop (Interval: {interval}s). Press Ctrl+C to stop.")
        try:
            while True:
                self.poll_and_propagate()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Polling stopped by user.")



if __name__ == "__main__":
    from graph_writer import writer_lease
    with writer_lease():
        kg = KGBuilder()
        try:
            kg.build_initial_graph()
            kg.poll_and_propagate()
        finally:
            kg.close()
