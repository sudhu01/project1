"""
graphrag.py — Thin entrypoint for standalone GraphRAG inspection.
For the full interactive RCA session run: python pipeline/llm_chat.py
"""

import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from retriever import _make_graph, get_active_anomalies, build_context

if __name__ == "__main__":
    graph = _make_graph()
    anomalies = get_active_anomalies(graph)

    if not anomalies:
        print("\n✅ All services healthy. No active alerts.\n")
    else:
        alerted = anomalies[0]["name"]
        print(f"\n[GraphRAG] Alert detected on: [{alerted}] — retrieving context...\n")
        ctx = build_context(graph, alerted)
        print(json.dumps(ctx, indent=2))

    graph.close()
