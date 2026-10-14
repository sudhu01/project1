"""
collect.py — Live anomaly polling loop.
Polls Prometheus every 30s, updates Neo4j, prints alert banners on incidents.
"""

import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from kg_builder import KGBuilder
from graph_writer import writer_lease

POLL_INTERVAL = 30

def _run():
    kg = KGBuilder()
    kg.build()

    print(f"\n🚀 Live polling started (every {POLL_INTERVAL}s) — Press Ctrl+C to stop\n")

    was_anomalous = False

    try:
        while True:
            anomalies = kg.poll_all_services()

            if anomalies:
                print(f"\n{'═'*60}")
                print(f"  🚨 ANOMALOUS SERVICES: {anomalies}")
                print(f"     → Run 'python pipeline/llm_chat.py' to investigate")
                print(f"{'═'*60}\n")
                was_anomalous = True
            else:
                if was_anomalous:
                    print(f"\n{'═'*60}")
                    print("  💚 SYSTEM RECOVERED — All services healthy")
                    print(f"{'═'*60}\n")
                    was_anomalous = False
                else:
                    print("  No confirmed anomaly; check the monitor for missing telemetry.")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n  Polling stopped.")
        kg.close()

def run():
    with writer_lease():
        _run()


if __name__ == "__main__":
    run()
