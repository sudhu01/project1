"""
monitor.py — Live system health monitor.
Polls Neo4j + Prometheus every 30s and prints a clean status table.
Prints an alert when anomalies are detected.
"""

import time
import warnings
import os
import sys

warnings.filterwarnings("ignore")

from kg_builder import KGBuilder
from graph_writer import writer_lease

POLL = 30


def _clear():
    os.system("clear")


def _status_line(name, status, cpu, mem, anomaly):
    flag   = "⚠" if anomaly else " "
    cpu_s  = f"{cpu:.1f}%" if cpu is not None else "  — "
    mem_s  = f"{mem:.0f}MB" if mem is not None else "  — "
    sta_s  = status or "unknown"
    return f"  {flag}  {name:<20} {sta_s:<10} {cpu_s:>8}  {mem_s:>8}"


def _render(services, anomalies, cycle):
    _clear()
    print(f"\n  graph-rca monitor   cycle #{cycle}   (ctrl+c to stop)\n")
    print(f"  {'service':<22} {'status':<10} {'cpu':>8}  {'memory':>8}")
    print(f"  {'─'*56}")

    for svc in services:
        print(_status_line(
            svc.get("name"), svc.get("status"),
            svc.get("cpu_usage"), svc.get("memory_usage"),
            svc.get("anomaly_flag"),
        ))

    print(f"  {'─'*56}")

    if anomalies:
        print(f"\n  ⚠  alert: {', '.join(anomalies)}")
        print(f"     run 'python pipeline/llm_chat.py' to investigate\n")
    elif any(svc.get("status") != "healthy" for svc in services) or not services:
        print("\n  health unknown, inspect missing telemetry\n")
    else:
        print("\n  all observed services healthy\n")


def _run():
    kg    = KGBuilder()
    kg.build()
    cycle = 0

    try:
        while True:
            cycle += 1
            # First poll is a grace period — services may still be warming up
            grace     = (cycle == 1)
            anomalies = kg.poll_and_propagate(grace=grace)

            # Fetch current node states for display
            try:
                services = kg.graph.query("""
                    MATCH (n)
                    WHERE n.role IS NOT NULL
                    RETURN n.name AS name, n.status AS status,
                           n.cpu_usage AS cpu_usage,
                           n.memory_usage AS memory_usage,
                           n.anomaly_flag AS anomaly_flag
                    ORDER BY n.name
                """)
            except Exception:
                services = []

            _render(services, anomalies, cycle)
            time.sleep(POLL)

    except KeyboardInterrupt:
        print("\n  monitor stopped.\n")
        try:
            kg.close()
        except Exception:
            pass


def run():
    with writer_lease():
        _run()


if __name__ == "__main__":
    run()
