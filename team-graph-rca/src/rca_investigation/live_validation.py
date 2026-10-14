"""Bounded readiness and workload proof, independent of fault injection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import monotonic, sleep
import requests
import yaml
from .dataset import atomic_json
from .workload import SockShopWorkload, WorkloadSpec


def pin_images(root, runner=subprocess.run):
    root = Path(root)
    docker = shutil.which('docker')
    config = yaml.safe_load((root/'docker-compose.yml').read_text(encoding='utf-8'))
    images = {}
    overrides = {'services': {}}
    for service, spec in config['services'].items():
        # The host's cached Mongo 3.4 extraction is damaged. Record the tested
        # replacement as an explicit experiment override, preserving legacy Compose.
        image = 'mongo:3.6.23' if service in ('carts-db','orders-db') else spec['image']
        result = runner([docker,'image','inspect',image],capture_output=True,text=True,timeout=10,shell=False)
        if result.returncode != 0:
            raise RuntimeError(f'image unavailable for {service}')
        row = json.loads(result.stdout)[0]
        digests = row.get('RepoDigests') or []
        if not digests:
            raise RuntimeError(f'image digest unavailable for {service}')
        images[service] = {'original_image':spec['image'], 'resolved_image':image, 'image_id':row['Id'], 'pinned_image':digests[0]}
        overrides['services'][service] = {'image':digests[0], 'cpus':2.0, 'mem_limit':'512m'}
    overrides['services']['neo4j'].update(mem_limit='1g', environment={
        'NEO4J_server_memory_heap_initial__size':'256m',
        'NEO4J_server_memory_heap_max__size':'512m',
        'NEO4J_server_memory_pagecache_size':'128m'})
    # Mongo 3.4 otherwise sizes its cache from host rather than this lab quota.
    for service in ('carts-db','orders-db'):
        overrides['services'][service]['command'] = ['mongod','--wiredTigerCacheSizeGB','1']
        overrides['services'][service]['mem_limit'] = '1536m'
    destination = root/'configs/lab_pinned.yaml'
    destination.write_text(yaml.safe_dump(overrides,sort_keys=True),encoding='utf-8')
    atomic_json(root/'runs/integration_v1/image-manifest.json', images)
    return images


def readiness(timeout_seconds=180):
    endpoints = {'catalogue':'http://localhost/catalogue', 'prometheus':'http://localhost:9090/-/ready',
                 'loki':'http://localhost:3100/ready', 'neo4j':'http://localhost:7474'}
    deadline = monotonic()+timeout_seconds
    status = {}
    while monotonic() < deadline:
        for name,url in endpoints.items():
            try:
                response = requests.get(url,timeout=3)
                ok = response.status_code == 200
                if name == 'catalogue':
                    ok &= isinstance(response.json(),list) and len(response.json()) > 0
                status[name] = {'ready':bool(ok),'status_code':response.status_code}
            except (requests.RequestException,ValueError):
                status[name] = {'ready':False}
        if all(record['ready'] for record in status.values()):
            return status
        sleep(3)
    raise RuntimeError('readiness timeout: '+json.dumps(status))


def verify_live(root):
    root = Path(root)
    report = {'passed':False,'timestamp':datetime.now(timezone.utc).isoformat()}
    workload = SockShopWorkload()
    try:
        report['readiness'] = readiness()
        report['login'] = workload.prepare()
        report['catalogue_orders_workload'] = workload.run(WorkloadSpec(1101,request_count=12,interval_seconds=.5))
        if report['catalogue_orders_workload']['successes'] != 12:
            raise RuntimeError('healthy workload failed')
        catalogue = workload.session.get('http://localhost/catalogue',timeout=5).json()
        affordable = [row for row in catalogue if 0 < float(row['price']) < 80]
        if not affordable:
            raise RuntimeError('no item within the demo payment limit')
        item = min(affordable,key=lambda row:float(row['price']))['id']
        cleared = workload.session.delete('http://localhost/cart',timeout=5)
        if cleared.status_code not in (200,202,204):
            raise RuntimeError('demo cart could not be cleared before checkout')
        cart = workload.session.post('http://localhost/cart',json={'id':item},timeout=5)
        order = workload.session.post('http://localhost/orders',json={},timeout=15) if cart.status_code == 201 else None
        report['checkout'] = {'cart_status':cart.status_code,'order_status':None if order is None else order.status_code}
        if order is None or order.status_code != 201:
            raise RuntimeError('order creation route not verified')
        targets = requests.get('http://localhost:9090/api/v1/targets',timeout=5).json()
        report['prometheus_targets'] = [{k:t.get(k) for k in ('labels','health','lastError','scrapeUrl')}
                                         for t in targets['data']['activeTargets']]
        # Front-end logs each API request. Verify the exact resolved container label.
        labels = requests.get('http://localhost:3100/loki/api/v1/label/container/values',timeout=5).json()
        candidates = [v for v in labels.get('data',[]) if v in ('/rca-lab-front-end-1','rca-lab-front-end-1')]
        if len(candidates) != 1:
            raise RuntimeError('front-end Loki label not observed')
        end = int(datetime.now(timezone.utc).timestamp()*1e9)
        logs = requests.get('http://localhost:3100/loki/api/v1/query_range',params={
            'query':'{container='+json.dumps(candidates[0])+'}', 'start':str(end-180_000_000_000),
            'end':str(end),'limit':20},timeout=5).json()
        count = sum(len(row['values']) for row in logs.get('data',{}).get('result',[]))
        report['loki_ingestion'] = {'container_label':candidates[0],'lines_observed':count}
        if count == 0:
            raise RuntimeError('no front-end log reached Loki')
        report['passed'] = True
    except (RuntimeError,requests.RequestException,ValueError,KeyError) as error:
        report['error'] = str(error)
    finally:
        workload.close()
        atomic_json(root/'runs/integration_v1/live-validation.json',report)
    return report
