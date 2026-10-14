"""
llm_chat.py — RCA Agent chat session.
"""

import os
import json
import time
import warnings
import threading
import textwrap
from pathlib import Path
from datetime import datetime, timezone

import logging
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from retriever import _make_graph, get_active_anomalies, build_context

console = Console(highlight=False)
WIDTH   = 60   # wrap width for agent replies


# ─── LLM loader ──────────────────────────────────────────────────────────────

def _load_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        import google.generativeai as genai
        key   = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not key or key.startswith("your_"):
            raise ValueError("GEMINI_API_KEY not set in pipeline/.env")
        genai.configure(api_key=key)
        return ("gemini", model)   # store model name, not object
    if provider == "groq":
        from groq import Groq
        key   = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not key or key.startswith("your_"):
            raise ValueError("GROQ_API_KEY not set in pipeline/.env")
        return ("groq", Groq(api_key=key), model)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")


def _call(bundle, system_msg, history, user_msg):
    if bundle[0] == "gemini":
        import google.generativeai as genai
        # Rebuild model with current system_instruction every call
        # so context refresh mid-chat is always honoured
        model = genai.GenerativeModel(
            bundle[1],
            system_instruction=system_msg
        )
        turns = []
        for t in history:
            turns.append({"role": "user",  "parts": [t["user"]]})
            turns.append({"role": "model", "parts": [t["assistant"]]})
        chat = model.start_chat(history=turns)
        return chat.send_message(user_msg).text.strip()
    if bundle[0] == "groq":
        msgs = [{"role": "system", "content": system_msg}]
        for t in history:
            msgs.append({"role": "user",      "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        msgs.append({"role": "user", "content": user_msg})
        r = bundle[1].chat.completions.create(
            model=bundle[2], messages=msgs, max_tokens=400
        )
        return r.choices[0].message.content.strip()


# ─── Context builder (Lean & Focused) ──────────────────────────────────────────

def _make_system_msg(ctx, healthy_mode=False, resolved=False):
    if resolved:
        return (
            "You are a concise RCA assistant for the Sock Shop microservices system.\n"
            "STATUS: RESOLVED. All services have recovered and are operating normally.\n"
            "If the user asks about the previous incident, confirm that the fault was repaired and the system is healthy.\n"
            "Keep answers short and conversational (1-3 sentences)."
        )

    if healthy_mode:
        services = ctx.get("all_services", [])
        names = ", ".join(s["name"] for s in services) if services else "all services"
        return (
            "You are a concise RCA assistant for the Sock Shop microservices system.\n"
            f"STATUS: HEALTHY. All 13 services are healthy ({names}).\n"
            "Answer questions about system architecture, dependencies, and health from this context.\n"
            "Keep answers short and conversational (1-3 sentences)."
        )

    inc   = ctx["incident"]
    anoms = ctx["anomalous_nodes"]
    logs  = ctx["logs"]
    trav  = ctx["traversal"]

    root  = inc['candidate_root_cause']
    atype = inc['anomaly_type']

    # Compact propagation summary
    chain = " → ".join(trav.get("propagation_chain", []))
    if not chain and anoms:
        chain = f"{root} (isolated)"

    # Only smoking gun logs (top 2 lines max per anomalous service)
    log_lines = []
    for svc, lines in logs.items():
        if lines:
            log_lines.append(f"{svc}: {lines[0][:140]}")
    log_summary = "\n  ".join(log_lines) if log_lines else "no error logs found"

    return (
        "You are a concise Root Cause Analysis assistant for the Sock Shop microservices system.\n"
        "Answer ONLY from the incident evidence below. Keep answers short, direct, and conversational (under 3 sentences).\n\n"
        f"[INCIDENT EVIDENCE]\n"
        f"• Root Cause: {root} ({atype})\n"
        f"• Traversal Path: {chain}\n"
        f"• Hops: {inc['hops_traversed']}\n"
        f"• Evidence Logs:\n  {log_summary}\n"
    )


# ─── Print helpers ────────────────────────────────────────────────────────────

def _agent_print(text):
    """Render Agent reply with markdown support."""
    console.print()
    md = Markdown(text)
    console.print("[bold]Agent:[/bold]", end=" ")
    console.print(md)
    console.print()


def _alert_print(services):
    console.print(f"\n[bold yellow]⚠  ALERT:[/bold yellow] Incident detected on [bold]{', '.join(services)}[/bold] — context updated\n")


def _resolved_print():
    console.print(f"\n[bold green]✅  RESOLVED:[/bold green] System recovered — all services healthy. Context updated\n")


def _header(status_line):
    console.print()
    console.rule("━", style="white")
    console.print(f"[bold]        RCA AGENT — ONLINE[/bold]")
    console.rule("━", style="white")
    console.print()


# ─── Watcher thread ───────────────────────────────────────────────────────────

class _Watcher(threading.Thread):
    def __init__(self, graph, known, on_alert, on_resolved):
        super().__init__(daemon=True)
        self.graph       = graph
        self.known       = set(known)
        self.on_alert    = on_alert
        self.on_resolved = on_resolved
        self._stop       = threading.Event()

    def run(self):
        while not self._stop.wait(10):
            try:
                rows = get_active_anomalies(self.graph)
                now  = {r["name"] for r in rows}
                new  = now - self.known

                if new:
                    self.known = now
                    self.on_alert(new, rows)
                elif not now and self.known:
                    # Anomaly was cleared!
                    self.known = set()
                    self.on_resolved()
            except Exception:
                pass

    def stop(self):
        self._stop.set()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    graph     = _make_graph()
    anomalies = get_active_anomalies(graph)

    healthy_mode  = not anomalies
    sys_msg_holder = [None]
    ctx            = {}

    if healthy_mode:
        try:
            services = graph.query("""
                MATCH (n) WHERE n.role IS NOT NULL
                RETURN n.name AS name, n.role AS role, n.status AS status
                ORDER BY n.name
            """)
        except Exception:
            services = []
        ctx = {"all_services": services}
        sys_msg_holder[0] = _make_system_msg(ctx, healthy_mode=True)

        greeting = (
            "Hey! I'm actively monitoring your cloud system. "
            f"All **{len(services)} services** are healthy right now.\n\n"
            "What would you like to know?"
        )
    else:
        alerted = anomalies[0]["name"]
        console.print(f"\n  retrieving context for [{alerted}]...")
        ctx = build_context(graph, alerted)
        sys_msg_holder[0] = _make_system_msg(ctx)
        root  = ctx["incident"]["candidate_root_cause"]
        atype = ctx["incident"]["anomaly_type"]
        hops  = ctx["incident"]["hops_traversed"]

        greeting = (
            f"I've detected an incident. **{root.upper()}** is the root cause candidate "
            f"({atype}), identified after **{hops} hop{'s' if hops != 1 else ''}** of graph traversal.\n\n"
            "What would you like to know?"
        )

    try:
        llm = _load_llm()
    except ValueError as e:
        console.print(f"\n[red]  error:[/red] {e}\n")
        return

    history = []

    def _on_new_alert(new_services, all_rows):
        alerted = list(new_services)[0]
        new_ctx = build_context(graph, alerted)
        sys_msg_holder[0] = _make_system_msg(new_ctx)
        _alert_print(new_services)

    def _on_resolved():
        sys_msg_holder[0] = _make_system_msg({}, resolved=True)
        _resolved_print()

    known = {a["name"] for a in anomalies}
    watcher = _Watcher(graph, known, _on_new_alert, _on_resolved)
    watcher.start()

    _header("online")
    _agent_print(greeting)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        reply = _call(llm, sys_msg_holder[0], history, user_input)
        _agent_print(reply)
        history.append({"user": user_input, "assistant": reply})

    watcher.stop()

    if history:
        out = Path(__file__).parent / "../data/chat_sessions.json"
        out.parent.mkdir(exist_ok=True)
        sessions = []
        if out.exists():
            try:
                sessions = json.loads(out.read_text())
            except Exception:
                pass
        sessions.append({
            "session_id": int(time.time()),
            "context":    sys_msg_holder[0],
            "turns":      history,
        })
        out.write_text(json.dumps(sessions, indent=2))
        n = len(history)
        console.print(f"\n  {n} turn{'s' if n != 1 else ''} saved → data/chat_sessions.json\n")
    else:
        console.print()

    try:
        graph.close()
    except Exception:
        pass


if __name__ == "__main__":
    run()
