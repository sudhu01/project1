# Local development environment

Steps 3 through 4.2 of the small simulator execution plan were completed on 2026-09-13.

- Python: CPython 3.12.13, 64-bit
- uv: 0.11.30
- PyTorch: 2.14.0+cpu
- Gymnasium: 1.3.0

`uv.lock` records the complete resolved dependency set. Use locked commands after the initial setup:

```powershell
uv sync --locked
uv run --locked pytest -q
```

`rca_sim.graph` contains the dependency DAG generator and graph queries. Edges
point from callers to dependencies. Generated graphs use randomized public
service IDs even though construction starts in a temporary topological order.

`rca_sim.world` defines and samples the private incident hypothesis. Cause
service, fault type, and workload use independent uniform priors. The entry
alert does not filter or otherwise change that prior.
