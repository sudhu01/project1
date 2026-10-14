"""Explicit engineering recovery check; rejected incident stays rejected."""
from datetime import datetime,timezone
import json
from pathlib import Path
from rca_investigation.dataset import atomic_json
from rca_investigation.faults import LabDriver
from rca_investigation.topology import sockshop_topology
from rca_investigation.workload import SockShopWorkload,WorkloadSpec

root=Path.cwd()
audit=root/'runs/integration_v1/captures/sockshop_pilot_v1'
marker=audit/'blocked-lab.json'
blocked=json.loads(marker.read_text(encoding='utf-8'))
if (audit/'capture.lock').exists(): raise RuntimeError('capture lease still active')
attempt=json.loads((audit/'attempts'/f"{blocked['incident_id']}.json").read_text(encoding='utf-8'))
driver=LabDriver(sockshop_topology(),root)
ids=driver.resolve()
if ids!=attempt['registry']: raise RuntimeError('recorded container identities changed')
states=driver.states()
if not all(value=='running' for value in states.values()): raise RuntimeError('lab resources not all running')
workload=SockShopWorkload()
try:
    login=workload.prepare()
    check=workload.run(WorkloadSpec(4404,request_count=12,interval_seconds=1))
    if check['successes']!=12: raise RuntimeError('recovery workload failed')
finally: workload.close()
atomic_json(audit/'later-recovery'/f"{blocked['incident_id']}.json",{'timestamp':datetime.now(timezone.utc).isoformat(),
    'blocked_incident':blocked,'states':states,'login':login,'workload':check,'confirmed':True,
    'original_attempt_remains_rejected':True})
schedule_path=audit/'schedule.json'
schedule=json.loads(schedule_path.read_text(encoding='utf-8'))
if blocked['incident_id'] not in schedule['rejected']:
    schedule['rejected'].append(blocked['incident_id'])
    atomic_json(schedule_path,schedule)
marker.unlink()
print('Exact container identities and recovery verified. Rejected attempt retained.')
