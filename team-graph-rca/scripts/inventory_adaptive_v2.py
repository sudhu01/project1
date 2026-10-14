from pathlib import Path
import json
from rca_investigation.capabilities import inventory, save_inventory

root = Path(__file__).resolve().parents[1]
report = inventory(root)
save_inventory(root, report)
print(json.dumps({'passed': report['passed'], 'errors': report['errors'],
                  'resources': {key: {'running': item['running'], 'capabilities': item['capabilities'],
                                      'metric_names': item['metric_names']} for key, item in report['resources'].items()}}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
