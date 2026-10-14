import requests
from neo4j import GraphDatabase

def check_sockshop():
    try:
        r = requests.get("http://localhost:80", timeout=5)
        if r.status_code == 200:
            print("✅ Sock Shop UI (port 80): Reachable (HTTP 200)")
        else:
            print(f"⚠️ Sock Shop UI (port 80): HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ Sock Shop UI (port 80): {e}")

def check_prometheus():
    try:
        r = requests.get("http://localhost:9090/api/v1/query", params={"query": "up"}, timeout=5)
        results = r.json().get("data", {}).get("result", [])
        print(f"✅ Prometheus (port 9090): {len(results)} targets reporting UP")
    except Exception as e:
        print(f"❌ Prometheus (port 9090): {e}")

def check_loki():
    try:
        r = requests.get("http://localhost:3100/ready", timeout=5)
        print(f"✅ Loki (port 3100): Status '{r.text.strip()}'")
    except Exception as e:
        print(f"❌ Loki (port 3100): {e}")

def check_neo4j():
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123")
        )
        with driver.session() as s:
            result = s.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
        print(f"✅ Neo4j (port 7474/7687): Connected, {count} nodes in KG")
        driver.close()
    except Exception as e:
        print(f"❌ Neo4j (port 7474/7687): {e}")

if __name__ == "__main__":
    print("=== RCA ENVIRONMENT VERIFICATION ===\n")
    check_sockshop()
    check_prometheus()
    check_loki()
    check_neo4j()
    print("\n====================================")
