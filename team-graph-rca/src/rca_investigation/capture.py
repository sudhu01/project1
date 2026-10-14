"""Exclusive, persisted capture lifecycle with mandatory recovery."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
import re
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4
import numpy as np
import yaml
from .backends.docker import DockerBackend
from .catalogue import catalogue_hash
from .contracts import IncidentPublic, IncidentPrivate, canonical
from .dataset import atomic_json, write_bundle


class RecoveryFailed(RuntimeError):
    pass


@contextmanager
def capture_lease(root):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'blocked-lab.json').exists():
        raise RecoveryFailed('unconfirmed prior recovery blocks the lab')
    target = root/'capture.lock'
    try:
        descriptor = os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError('another capture or an interrupted lease owns this lab') from error
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump({'pid':os.getpid(),'project':'rca-lab'},stream)
        yield
    finally:
        target.unlink()


class CaptureController:
    def __init__(self, config, topology, specs, driver, workload, dataset, audit, *, backend_factory=DockerBackend,
                 window_seconds=60, baseline_seconds=60, interval_seconds=2, sleeper=sleep, timer=monotonic, now=None):
        if window_seconds != 60 or baseline_seconds != 60:
            raise ValueError('production capture requires 60 second baseline and observation windows')
        if type(interval_seconds) is not int or interval_seconds not in (1,2):
            raise ValueError('only verified one- or two-second request intervals are supported')
        self.config,self.topology,self.specs = config,topology,tuple(specs)
        self.driver,self.workload = driver,workload
        self.dataset,self.audit = Path(dataset),Path(audit)
        self.backend_factory = backend_factory
        self.window_seconds,self.baseline_seconds = window_seconds,baseline_seconds
        self.interval_seconds = interval_seconds
        self.sleep,self.timer = sleeper,timer
        self.now = now or (lambda:datetime.now(timezone.utc))

    def _state(self, receipt, state):
        receipt['status'] = state
        receipt['transitions'].append({'state':state,'timestamp':self.now().isoformat()})
        atomic_json(self.audit/'attempts'/f"{receipt['incident_id']}.json",receipt)

    def _expected(self, scenario):
        return {r:'exited' if r == scenario else 'running' for r in self.topology.resources}

    def _window(self, seconds, seed, expected, backend=None, *, deadline=None):
        started = self.timer()
        deadline = started+seconds if deadline is None else deadline
        rng = np.random.default_rng(seed)
        requests,states = [],[]
        packets = ()
        if backend is not None:
            packets = tuple(backend.execute(spec) for spec in self.specs)
        index = 0
        while self.timer() < deadline:
            observed = self.driver.states()
            if observed != expected:
                raise RuntimeError('resource state changed during capture window')
            states.append({'sample_time':self.now().isoformat(),'states':observed})
            result = self.workload.request('/catalogue' if index % 2 == 0 else '/orders')
            requests.append(result)
            index += 1
            self.sleep(min(self.interval_seconds*float(rng.uniform(.9,1.1)),max(0,deadline-self.timer())))
        elapsed=self.timer()-started
        return {'seed':seed,'requests':requests,'states':states,'request_count':len(requests),
                'interval_seconds':self.interval_seconds,'concurrency':1,'elapsed_seconds':elapsed,
                'achieved_requests_per_second':len(requests)/max(elapsed,1e-9)},packets

    def _ready(self, timeout=90):
        deadline = self.timer()+timeout
        observed,checks={},[]
        while self.timer() < deadline:
            observed=self.driver.states()
            if all(v == 'running' for v in observed.values()):
                checks = [self.workload.request(route) for route in ('/catalogue','/orders')]
                if all(record['success'] for record in checks):
                    return {'confirmed':True,'requests':checks,'timestamp':self.now().isoformat()}
            self.sleep(2)
        raise RuntimeError('lab/workload readiness failed: '+json.dumps({'states':observed,'requests':checks}))

    def capture(self, scenario, *, session, lab_batch, seed):
        if scenario not in ('catalogue-db','orders','NO_FAULT'):
            raise ValueError('unapproved capture scenario')
        values = self.config.values
        incident_id = str(uuid4())
        receipt = {'incident_id':incident_id,'physical_episode':str(uuid4()),'capture_session':session,
                   'lab_batch':lab_batch,'scenario':scenario,'schedule_seed':seed,'transitions':[],
                   'recovery_confirmed':False,'config_hash':self.config.hash}
        injected = False
        failure = None
        packets = ()
        public = None
        with capture_lease(self.audit):
            self._state(receipt,'CREATED')
            try:
                receipt['registry'] = self.driver.resolve()
                receipt['initial_readiness'] = self._ready()
                baseline,_ = self._window(self.baseline_seconds,seed,self._expected('NO_FAULT'))
                if not baseline['requests'] or not all(r['success'] for r in baseline['requests']):
                    raise RuntimeError('baseline workload degraded')
                receipt['baseline'] = baseline
                self._state(receipt,'BASELINE_READY')
                if scenario != 'NO_FAULT':
                    self._state(receipt,'INJECTION_REQUESTED')
                    atomic_json(self.audit/'blocked-lab.json',{'incident_id':incident_id,'resource':scenario,
                                                             'container_id':receipt['registry'][scenario]})
                    injected = True  # Recovery is mandatory even if dispatch raises.
                    receipt['injector'] = self.driver.inject_stop(scenario)
                    physical = self.driver.state(scenario)
                    if physical['state'] not in ('exited','dead'):
                        raise RuntimeError('stop effect not confirmed')
                    route = '/catalogue' if scenario == 'catalogue-db' else '/orders'
                    symptoms = [self.workload.request(route) for _ in range(2)]
                    if any(r['success'] for r in symptoms):
                        raise RuntimeError('stop did not degrade the relevant workload')
                    receipt['effect'] = {'confirmed':True,'physical':physical,'requests':symptoms}
                else:
                    receipt['injector'] = {'mechanism':'healthy','dispatched':False}
                    receipt['effect'] = {'confirmed':True,'readiness':self._ready()}
                self._state(receipt,'EFFECT_CONFIRMED')
                start = self.now()
                window_deadline=self.timer()+self.window_seconds
                public = IncidentPublic(incident_id,self.topology.hash,catalogue_hash(self.specs),'front-end',
                    values['entry_protocol'],start.isoformat(),(start+timedelta(seconds=self.window_seconds)).isoformat(),
                    values['episode']['budget'],values['episode']['max_probes'],values['objective']['lambda_cost'])
                backend = self.backend_factory(self.specs)
                try:
                    backend.begin(public)
                    observed,packets = self._window(self.window_seconds,seed+1,self._expected(scenario),backend,deadline=window_deadline)
                    if self.driver.states()!=self._expected(scenario):
                        raise RuntimeError('resource state changed at the observation boundary')
                    if any(p.availability != 'OK' for p in packets):
                        raise RuntimeError('capture has unavailable required packets')
                    route = '/catalogue' if scenario == 'catalogue-db' else '/orders'
                    relevant = [r for r in observed['requests'] if r['route'] == route]
                    if not relevant or (scenario == 'NO_FAULT' and not all(r['success'] for r in observed['requests'])):
                        raise RuntimeError('healthy observation workload degraded')
                    if scenario != 'NO_FAULT' and any(r['success'] for r in relevant):
                        raise RuntimeError('fault symptom did not persist through observation window')
                finally:
                    backend.close()
                receipt['observed_window'] = observed
                self._state(receipt,'WINDOW_CAPTURED')
            except BaseException as error:
                failure = error
                receipt['error'] = type(error).__name__+': '+str(error)
            finally:
                try:
                    if injected:
                        receipt['recovery_command'] = self.driver.recover(scenario)
                    receipt['recovery'] = self._ready()
                    receipt['recovery_confirmed'] = True
                    (self.audit/'blocked-lab.json').unlink(missing_ok=True)
                    self._state(receipt,'RECOVERED')
                except BaseException as recovery_error:
                    receipt['recovery_error'] = str(recovery_error)
                    atomic_json(self.audit/'blocked-lab.json',{'incident_id':incident_id,'resource':scenario,'error':str(recovery_error)})
                    self._state(receipt,'REJECTED')
                    raise RecoveryFailed('lab recovery unconfirmed; further captures blocked') from recovery_error
            if failure is not None:
                self._state(receipt,'REJECTED')
                if isinstance(failure,(KeyboardInterrupt,SystemExit)):
                    raise failure
                return receipt
            private = IncidentPrivate(incident_id,() if scenario == 'NO_FAULT' else (scenario,),
                'healthy' if scenario == 'NO_FAULT' else 'container_stopped',canonical(receipt['injector']),
                canonical(receipt['effect']),canonical(receipt['recovery']))
            receipt['status'] = 'ACCEPTED'
            try:
                receipt['bundle'] = str(write_bundle(self.dataset,public,self.topology,packets,private,receipt))
            except (OSError, ValueError) as error:
                receipt['error'] = str(error)
                self._state(receipt,'REJECTED')
                return receipt
            self._state(receipt,'ACCEPTED')
            atomic_json(self.audit/'attempts'/f'{incident_id}.json',receipt)
            return receipt


def load_schedule(path):
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if type(payload) is not dict or set(payload) != {'dataset','seed','sessions'}:
        raise ValueError('invalid capture schedule fields')
    if type(payload['dataset']) is not str or not Path(payload['dataset']).parts or Path(payload['dataset']).parts[0]!='datasets' or Path(payload['dataset']).is_absolute() or '..' in Path(payload['dataset']).parts:
        raise ValueError('dataset must stay within the lab workspace')
    if type(payload['seed']) is not int or payload['seed'] < 0 or type(payload['sessions']) is not list or not payload['sessions']:
        raise ValueError('invalid schedule seed or sessions')
    ids = set()
    for session in payload['sessions']:
        if type(session) is not dict or not {'id','per_class','role'} <= set(session) or set(session)-{'id','per_class','role','interval_seconds'} or type(session['id']) is not str or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',session['id']) or session['id'] in ids:
            raise ValueError('invalid or duplicate capture session')
        if session['role'] not in ('train','validation','test') or type(session['per_class']) is not int or not 1 <= session['per_class'] <= 20:
            raise ValueError('bounded class counts and reserved split roles required')
        ids.add(session['id'])
        if type(session.get('interval_seconds',2)) is not int or session.get('interval_seconds',2) not in (1,2):
            raise ValueError('unverified workload interval')
    if sum(s['per_class']*3 for s in payload['sessions']) > 150:
        raise ValueError('capture schedule exceeds 150 physical episodes')
    return payload


def run_schedule(root, schedule_path, config_path):
    from .config import load_config
    from .topology import sockshop_topology
    from .catalogue import build_catalogue
    from .contracts import digest
    from .faults import LabDriver
    from .workload import SockShopWorkload
    root = Path(root).resolve()
    schedule = load_schedule(schedule_path)
    dataset = root/schedule['dataset']
    if not dataset.resolve().is_relative_to(root):
        raise ValueError('dataset escapes workspace')
    audit = root/'runs/integration_v1/captures'/dataset.name
    audit.mkdir(parents=True,exist_ok=True)
    state_path = audit/'schedule.json'
    schedule_hash = digest(schedule)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['schedule_hash'] != schedule_hash:
            raise ValueError('resume schedule changed')
    else:
        state = {'schedule_hash':schedule_hash,'accepted':{},'rejected':[],'roles':{s['id']:s['role'] for s in schedule['sessions']}}
    topology = sockshop_topology()
    config = load_config(config_path,topology)
    workload = SockShopWorkload()
    controller = CaptureController(config,topology,build_catalogue(config,topology),LabDriver(topology,root),
                                   workload,dataset,audit)
    try:
        workload.prepare()
        rng = np.random.default_rng(schedule['seed'])
        for session in schedule['sessions']:
            controller.interval_seconds=session.get('interval_seconds',2)
            unfinished=any(session['id']+'/'+str(i) not in state['accepted'] for i in range(session['per_class']*3))
            context_path=audit/'sessions'/f"{session['id']}.json"
            if unfinished and not context_path.exists():
                atomic_json(context_path,{'id':session['id'],'role':session['role'],
                    'started_at':datetime.now(timezone.utc).isoformat(),'host':controller.driver.host_context(),
                    'request_interval_seconds':controller.interval_seconds,'concurrency':1,
                    'image_manifest':json.loads((root/'runs/integration_v1/image-manifest.json').read_text(encoding='utf-8'))})
            if unfinished:
                workload.close()
                workload=SockShopWorkload()
                workload.prepare()
                controller.workload=workload
            scenarios = ['catalogue-db','orders','NO_FAULT']*session['per_class']
            rng.shuffle(scenarios)
            for index,scenario in enumerate(scenarios):
                seed = int(rng.integers(0,2**31))
                slot = session['id']+'/'+str(index)
                if slot in state['accepted']:
                    continue
                workload.prepare()
                receipt = controller.capture(scenario,session=session['id'],lab_batch=dataset.name,seed=seed)
                if receipt['status'] != 'ACCEPTED':
                    state['rejected'].append(receipt['incident_id'])
                    atomic_json(state_path,state)
                    raise RuntimeError('capture rejected: '+receipt.get('error','unknown error'))
                state['accepted'][slot] = receipt['incident_id']
                atomic_json(state_path,state)
                print(json.dumps({'accepted':len(state['accepted']),'session':session['id'],'incident_id':receipt['incident_id']}),flush=True)
    finally:
        workload.close()
    state['complete']=True
    atomic_json(state_path,state)
    return {'passed':True,'physical_incidents':len(state['accepted']),'rejected':len(state['rejected']),
            'dataset':str(dataset),'schedule_hash':schedule_hash,'roles':state['roles']}
