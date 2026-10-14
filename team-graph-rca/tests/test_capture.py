from datetime import datetime, timedelta, timezone
import pytest
from rca_investigation.capture import CaptureController, RecoveryFailed, capture_lease
from rca_investigation.config import load_config
from rca_investigation.topology import sockshop_topology
from rca_investigation.catalogue import build_catalogue
from rca_investigation.contracts import ToolResult
from rca_investigation.dataset import load_private_label
import json
import yaml


class Clock:
    seconds = 0
    def timer(self): return self.seconds
    def sleep(self, seconds): self.seconds += seconds
    def now(self): return datetime(2026,10,6,tzinfo=timezone.utc)+timedelta(seconds=self.seconds)


class Driver:
    target = None
    recoveries = 0
    recovery_fails = False
    def resolve(self): return {r:r for r in sockshop_topology().resources}
    def states(self): return {r:'exited' if r == self.target else 'running' for r in sockshop_topology().resources}
    def state(self, r): return {'state':self.states()[r]}
    def inject_stop(self, r): self.target=r; return {'dispatched':True}
    def recover(self, r):
        self.recoveries += 1
        if self.recovery_fails: raise RuntimeError('failed recovery')
        self.target=None
        return {'started':True}


def setup(tmp_path, interruption=False, invisible=False):
    clock, driver = Clock(), Driver()
    class Workload:
        def request(self, route):
            if interruption and driver.target: raise KeyboardInterrupt()
            fails = driver.target == ('catalogue-db' if route == '/catalogue' else 'orders')
            return {'route':route,'success':not fails or invisible}
    class Backend:
        def __init__(self, specs): pass
        def begin(self, public): self.public=public
        def close(self): pass
        def execute(self, spec):
            p=self.public
            return ToolResult(spec.action_id,spec.resource_id,'OK',
                'STOPPED' if driver.target == spec.resource_id else 'RUNNING',
                p.window_start,p.window_end,clock.now().isoformat(),'mock',0)
    topology=sockshop_topology()
    config=load_config('configs/investigation_v1.yaml',topology)
    controller=CaptureController(config,topology,build_catalogue(config,topology),driver,Workload(),
        tmp_path/'dataset',tmp_path/'audit',backend_factory=Backend,
        sleeper=clock.sleep,timer=clock.timer,now=clock.now)
    return controller,driver


@pytest.mark.parametrize('scenario',['NO_FAULT','orders','catalogue-db'])
def test_capture_accepts_only_measured_recovered_incidents(tmp_path,scenario):
    controller,driver=setup(tmp_path)
    receipt=controller.capture(scenario,session='s1',lab_batch='b1',seed=11)
    assert receipt['status']=='ACCEPTED' and receipt['recovery_confirmed']
    assert driver.target is None
    assert len(receipt['baseline']['requests']) >= 25
    assert load_private_label(receipt['bundle']).resources == (() if scenario=='NO_FAULT' else (scenario,))


def test_interruption_always_recovers(tmp_path):
    controller,driver=setup(tmp_path,interruption=True)
    with pytest.raises(KeyboardInterrupt): controller.capture('orders',session='s',lab_batch='b',seed=1)
    assert driver.recoveries==1 and driver.target is None
    assert not (tmp_path/'audit/blocked-lab.json').exists()
    assert not (tmp_path/'dataset').exists()


def test_invisible_effect_rejected(tmp_path):
    controller,driver=setup(tmp_path,invisible=True)
    receipt=controller.capture('orders',session='s',lab_batch='b',seed=1)
    assert receipt['status']=='REJECTED' and driver.recoveries==1
    assert not (tmp_path/'dataset').exists()


def test_failed_recovery_blocks_next_capture(tmp_path):
    controller,driver=setup(tmp_path)
    driver.recovery_fails=True
    with pytest.raises(RecoveryFailed): controller.capture('orders',session='s',lab_batch='b',seed=1)
    with pytest.raises(RecoveryFailed): controller.capture('NO_FAULT',session='s',lab_batch='b',seed=2)
    assert driver.recoveries==1


def test_capture_exclusive(tmp_path):
    with capture_lease(tmp_path):
        with pytest.raises(RuntimeError):
            with capture_lease(tmp_path): pass


def test_schedule_resumes_slots_with_same_seeds_and_order(tmp_path,monkeypatch):
    import rca_investigation.capture as capture
    import rca_investigation.faults as faults
    import rca_investigation.workload as workload
    calls=[]
    class FakeDriver:
        def __init__(self,*args): pass
        def host_context(self): return {'NCPU':12}
    class FakeWorkload:
        def prepare(self): pass
        def close(self): pass
    class FakeController:
        def __init__(self,*args): self.driver=args[3]
        def capture(self,scenario,**kwargs):
            if len(calls)==2 and interrupted[0]:
                interrupted[0]=False
                raise KeyboardInterrupt()
            calls.append((scenario,kwargs['session'],kwargs['seed']))
            return {'status':'ACCEPTED','incident_id':f'fake-{len(calls)}'}
    monkeypatch.setattr(faults,'LabDriver',FakeDriver)
    monkeypatch.setattr(workload,'SockShopWorkload',FakeWorkload)
    monkeypatch.setattr(capture,'CaptureController',FakeController)
    schedule=tmp_path/'capture.yaml'
    schedule.write_text(yaml.safe_dump({'dataset':'datasets/test','seed':11,'sessions':[
        {'id':role,'per_class':1,'role':role,'interval_seconds':1} for role in ('train','validation','test')]}))
    manifest=tmp_path/'runs/integration_v1/image-manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    interrupted=[True]
    with pytest.raises(KeyboardInterrupt): capture.run_schedule(tmp_path,schedule,'configs/investigation_v1.yaml')
    assert len(calls)==2
    result=capture.run_schedule(tmp_path,schedule,'configs/investigation_v1.yaml')
    assert result['physical_incidents']==9 and len(calls)==9
    state=json.loads((tmp_path/'runs/integration_v1/captures/test/schedule.json').read_text())
    assert state['complete'] and len(set(state['accepted'].values()))==9
    capture.run_schedule(tmp_path,schedule,'configs/investigation_v1.yaml')
    assert len(calls)==9
