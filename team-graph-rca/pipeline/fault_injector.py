"""
fault_injector.py — Interactive fault injection for RCA demo.
"""

import subprocess
import time
import json
import os
from datetime import datetime, timezone

FAULT_LOG = os.path.join(os.path.dirname(__file__), "../data/fault_log.json")

FAULTS = {
    # ── Basic ────────────────────────────────────────────────────────────────
    "1": {
        "name": "db_crash",
        "category": "dependency_failure",
        "target": "catalogue-db",
        "container": "rca-project-catalogue-db-1",
        "inject": "docker stop rca-project-catalogue-db-1",
        "recover": "docker start rca-project-catalogue-db-1",
        "description": "Kill catalogue-db (database dependency crash)",
    },
    "2": {
        "name": "cpu_exhaustion",
        "category": "resource_exhaustion",
        "target": "payment",
        "container": "rca-project-payment-1",
        "inject": 'docker exec -d rca-project-payment-1 sh -c "while :; do :; done & while :; do :; done &"',
        "recover": "docker restart rca-project-payment-1",
        "description": "CPU burn on payment service",
    },
    "3": {
        "name": "service_crash",
        "category": "cascading_failure",
        "target": "orders",
        "container": "rca-project-orders-1",
        "inject": "docker stop rca-project-orders-1",
        "recover": "docker start rca-project-orders-1",
        "description": "Kill orders service (cascades to frontend)",
    },
    "4": {
        "name": "bad_deploy",
        "category": "config_deploy",
        "target": "payment",
        "container": "rca-project-payment-1",
        "inject": "docker stop rca-project-payment-1",
        "recover": "docker start rca-project-payment-1",
        "description": "Simulate bad deployment — stop payment",
    },
    # ── Deep / Hard to explain without LLM ──────────────────────────────────
    "5": {
        "name": "memory_leak",
        "category": "resource_exhaustion",
        "target": "carts",
        "container": "rca-project-carts-1",
        "inject": (
            'docker exec -d rca-project-carts-1 sh -c '
            '"python3 -c \\"import time; d=[]; '
            '[d.extend([b\'x\'*1024*1024]) or time.sleep(0.3) for _ in range(9999)]\\" '
            '|| dd if=/dev/zero bs=1M count=800 > /dev/null &"'
        ),
        "recover": "docker restart rca-project-carts-1",
        "description": "Memory leak in carts — gradual OOM, slow onset",
    },
    "6": {
        "name": "upstream_db_crash",
        "category": "upstream_dependency",
        "target": "orders-db",
        "container": "rca-project-orders-db-1",
        "inject": "docker stop rca-project-orders-db-1",
        "recover": "docker start rca-project-orders-db-1",
        "description": "Kill orders-db — orders times out, cascades to frontend",
    },
    "7": {
        "name": "user_service_crash",
        "category": "auth_failure",
        "target": "user",
        "container": "rca-project-user-1",
        "inject": "docker stop rca-project-user-1",
        "recover": "docker start rca-project-user-1",
        "description": "Kill user service — login/auth broken across entire shop",
    },
    "8": {
        "name": "multi_service_chaos",
        "category": "cascading_failure",
        "target": "catalogue-db + payment",
        "container": "rca-project-catalogue-db-1",
        "inject": (
            "docker stop rca-project-catalogue-db-1 && "
            'docker exec -d rca-project-payment-1 sh -c "while :; do :; done &"'
        ),
        "recover": (
            "docker start rca-project-catalogue-db-1 && "
            "docker restart rca-project-payment-1"
        ),
        "description": "Multi-fault chaos — DB crash + CPU storm simultaneously",
    },
    "9": {
        "name": "shipping_crash",
        "category": "downstream_failure",
        "target": "shipping",
        "container": "rca-project-shipping-1",
        "inject": "docker stop rca-project-shipping-1",
        "recover": "docker start rca-project-shipping-1",
        "description": "Kill shipping — orders complete but fulfilment silently breaks",
    },
}


def _log_fault(fault, injection_time):
    os.makedirs(os.path.dirname(FAULT_LOG), exist_ok=True)
    try:
        with open(FAULT_LOG) as f:
            log = json.load(f)
    except Exception:
        log = []

    entry = {
        "episode_id": len(log) + 1,
        "fault_name": fault["name"],
        "fault_category": fault["category"],
        "target_service": fault["target"],
        "injection_time": injection_time,
        "actual_root_cause": fault["target"],
        "description": fault["description"],
    }
    log.append(entry)
    with open(FAULT_LOG, "w") as f:
        json.dump(log, f, indent=2)
    return entry["episode_id"]


def _c(service_name):
    """Find the exact running or existing container name for a service."""
    try:
        cmd = ["docker", "ps", "-a", "--format", "{{.Names}}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            for name in res.stdout.strip().splitlines():
                if f"-{service_name}-" in name or name.endswith(f"-{service_name}") or name.endswith(f"-{service_name}-1"):
                    return name
    except Exception:
        pass
    return f"graph-rca-{service_name}-1"

def inject(fault_id):
    fault = FAULTS[fault_id]
    t = datetime.now(timezone.utc).isoformat()
    print(f"\n  injecting › {fault['description']}...")
    try:
        # Resolve target containers dynamically
        cmd = fault["inject"]
        for svc in ["catalogue-db", "payment", "orders", "carts", "orders-db", "user", "shipping"]:
            actual = _c(svc)
            cmd = cmd.replace(f"rca-project-{svc}-1", actual).replace(f"graph-rca-{svc}-1", actual)

        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ep = _log_fault(fault, t)
        print(f"  done › episode #{ep} logged  |  monitor will detect in ≤30s")
    except Exception as e:
        print(f"  error › {e}")


def recover(fault_id):
    fault = FAULTS[fault_id]
    print(f"\n  recovering › {fault['target']}...")
    try:
        cmd = fault["recover"]
        for svc in ["catalogue-db", "payment", "orders", "carts", "orders-db", "user", "shipping"]:
            actual = _c(svc)
            cmd = cmd.replace(f"rca-project-{svc}-1", actual).replace(f"graph-rca-{svc}-1", actual)

        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  done › {fault['target']} restarted")
    except Exception as e:
        print(f"  error › {e}")


def menu():
    print("\n  graph-rca fault injector\n")
    print("  ── basic ──────────────────────────────────────")
    for k in ["1", "2", "3", "4"]:
        v = FAULTS[k]
        print(f"  {k}  {v['description']}")
    print("\n  ── advanced ───────────────────────────────────")
    for k in ["5", "6", "7", "8", "9"]:
        v = FAULTS[k]
        print(f"  {k}  {v['description']}")
    print("\n  r  recover last fault")
    print("  q  quit")
    print("  ───────────────────────────────────────────────")


if __name__ == "__main__":
    last = None
    while True:
        menu()
        choice = input("\n  › ").strip().lower()
        if choice == "q":
            print()
            break
        elif choice == "r":
            if last:
                recover(last)
                last = None
            else:
                print("  no active fault to recover")
        elif choice in FAULTS:
            inject(choice)
            last = choice
        else:
            print("  invalid choice")
