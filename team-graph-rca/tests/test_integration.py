from dataclasses import replace
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import subprocess
import sys
import pytest
from rca_investigation.config import load_config
from rca_investigation.topology import sockshop_topology
from rca_investigation.catalogue import build_catalogue,catalogue_hash
from rca_investigation.contracts import IncidentPublic,IncidentPrivate,ToolResult,digest
from rca_investigation.dataset import atomic_json,write_bundle,freeze_scheduled_splits,read_split
from rca_investigation.experiment import fit_dataset,load_model,replay_environment
from rca_investigation.evaluate_decisions import evaluate_paths,paired_session_interval
from rca_investigation.policies.rules import RulePolicy


def make_dataset(root):
    topology=sockshop_topology()
    config=load_config('configs/investigation_v1.yaml',topology)
    specs=build_catalogue(config,topology)
    roles={f's{i}':role for i,role in enumerate(('train','validation','test'))}
    state={'complete':True,'roles':roles,'accepted':{},'schedule_hash':'synthetic-test'}
    for session in roles:
        for i,scenario in enumerate(config.values['diagnoses']):
            incident=f'{session}-{i}'
            public=IncidentPublic(incident,topology.hash,catalogue_hash(specs),'front-end','manual_frontend',
                '2026-10-06T10:00:00Z','2026-10-06T10:01:00Z',6,4,.05)
            packets=tuple(ToolResult(s.action_id,s.resource_id,'OK','STOPPED' if s.resource_id==scenario else 'RUNNING',
                public.window_start,public.window_end,public.window_end,'synthetic',0) for s in specs)
            private=IncidentPrivate(incident,() if scenario=='NO_FAULT' else (scenario,),
                'healthy' if scenario=='NO_FAULT' else 'container_stopped','{}','{}','{}')
            receipt={'status':'ACCEPTED','recovery_confirmed':True,'capture_session':session,'physical_episode':incident,
                     'synthetic_only':True}
            write_bundle(root,public,topology,packets,private,receipt)
            state['accepted'][incident]=incident
    schedule=root.parent/'schedule.json'
    atomic_json(schedule,state)
    freeze_scheduled_splits(root,schedule)
    return config,specs


def test_fitting_evaluation_and_application_trace_agree(tmp_path):
    dataset=tmp_path/'dataset'
    config,specs=make_dataset(dataset)
    model_path=tmp_path/'model.json'
    model=fit_dataset(dataset,config,specs,model_path)
    manifest,paths=read_split(dataset,'validation')
    restored=load_model(model_path,config,specs,manifest['hash'])
    assert restored.hash==model.hash
    summary,outcomes=evaluate_paths(paths,specs,model,config,RulePolicy('dependency-rule'))
    assert summary['accuracy']==1 and summary['mean_credits']==pytest.approx(5/3)
    interval=paired_session_interval(outcomes,outcomes,samples=10)
    assert interval['session_bootstrap_95']==[0,0]
    args=['investigate','--backend','replay','--incident',str(paths[0]),'--policy','dependency-rule',
          '--dataset',str(dataset),'--model',str(model_path)]
    cli_output,app_output=tmp_path/'cli.json',tmp_path/'app.json'
    for prefix,output in (([sys.executable,'-c','from rca_investigation.cli import main; raise SystemExit(main())'],cli_output),
                          ([sys.executable,'pipeline/investigate.py'],app_output)):
        result=subprocess.run([*prefix,*args,'--output',str(output)],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
    cli=json.loads(cli_output.read_text())
    app=json.loads(app_output.read_text())
    assert cli['result']['decision']==app['result']['decision']
    assert [t['action'] for t in cli['result']['trace']]==[t['action'] for t in app['result']['trace']]
    assert cli['explanation']['evidence_ids']==cli['result']['decision']['evidence_ids']
    assert len(cli['explanation']['evidence'])==cli['result']['probes']
    assert model.training_hashes and len(model.training_hashes)==3
    with pytest.raises(ValueError): freeze_scheduled_splits(dataset,tmp_path/'schedule.json')


def test_manifest_tampering_and_episode_overlap_rejected(tmp_path):
    dataset=tmp_path/'dataset'
    make_dataset(dataset)
    path=dataset/'splits.json'
    payload=json.loads(path.read_text())
    payload['splits']['train'].append('s2-0')
    atomic_json(path,payload)
    with pytest.raises(ValueError): read_split(dataset,'train')
    payload.pop('hash')
    payload['hash']=digest(payload)
    atomic_json(path,payload)
    with pytest.raises(ValueError): read_split(dataset,'train')


def test_reserved_comparison_cli_preserves_partial_and_completed_reports(tmp_path, monkeypatch):
    from rca_investigation import comparison
    from rca_investigation.cli import main
    from rca_investigation.train import runtime_hash
    dataset=tmp_path/'dataset'
    config,specs=make_dataset(dataset)
    model_path=tmp_path/'model.json'
    model=fit_dataset(dataset,config,specs,model_path)
    manifest,_=read_split(dataset,'train')
    frozen={'schema_version':1,'dataset':str(dataset.resolve()),'split_hash':manifest['hash'],
            'config_hash':config.hash,'model_hash':model.hash,'runtime_hash':runtime_hash(),
            'definitions':{'dependency-rule':{'kind':'rule','name':'dependency-rule'},
                           'stop-immediately':{'kind':'rule','name':'stop-immediately'}},
            'selected':'dependency-rule','checkpoint_hashes':{},
            'dependency_lock_hash':digest(Path('uv.lock').read_text(encoding='utf-8'))}
    frozen['hash']=digest(frozen)
    frozen_path,output=tmp_path/'frozen.json',tmp_path/'test-report.json'
    atomic_json(frozen_path,frozen)
    evaluate=comparison.evaluate_paths
    calls=[]

    def interrupt_second(*args, **kwargs):
        calls.append(args[4].name)
        if len(calls)==2:
            raise RuntimeError('simulated interruption after first policy')
        return evaluate(*args,**kwargs)

    monkeypatch.setattr(comparison,'evaluate_paths',interrupt_second)
    arguments=['compare','--split','test','--dataset',str(dataset),'--model',str(model_path),
               '--frozen-manifest',str(frozen_path),'--output',str(output)]
    assert main(arguments)==1
    partial=json.loads(output.read_text())
    assert not partial['passed'] and list(partial['summaries'])==['dependency-rule']
    assert output.with_suffix('.json.failure.json').exists()
    resumed=[]

    def record_resume(*args, **kwargs):
        resumed.append(args[4].name)
        return evaluate(*args,**kwargs)

    monkeypatch.setattr(comparison,'evaluate_paths',record_resume)
    assert main(arguments)==0
    assert resumed==['stop-immediately']
    completed=output.read_bytes()
    assert main(arguments)==1
    assert output.read_bytes()==completed
    # A frozen validation selection is final even if its dataset later becomes
    # unreadable. Refusal must precede reading/scoring or replacing the report.
    def forbidden_read(*args, **kwargs):
        raise AssertionError('frozen validation was reopened')
    monkeypatch.setattr(comparison,'read_split',forbidden_read)
    with pytest.raises(ValueError,match='already frozen'):
        comparison.compare_validation(dataset,specs,model,config,tmp_path,output,frozen_path)
    assert output.read_bytes()==completed
