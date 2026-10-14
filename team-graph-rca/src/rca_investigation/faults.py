"""Fixed injection commands owned by the capture controller, never a policy."""
from datetime import datetime, timezone
import json
import shutil
import subprocess
from .backends.docker import compose_rows, resolve_containers


class LabDriver:
    def __init__(self, topology, root, runner=subprocess.run):
        self.topology, self.root, self.runner = topology, root, runner
        self.docker = shutil.which('docker')
        if self.docker is None:
            raise RuntimeError('Docker unavailable')
        self.ids = {}

    def command(self, argv, timeout=15):
        result = self.runner([self.docker,*argv],cwd=self.root,capture_output=True,text=True,timeout=timeout,shell=False)
        receipt = {'argv':argv,'exit_code':result.returncode,'timestamp':datetime.now(timezone.utc).isoformat()}
        if result.returncode != 0:
            raise RuntimeError('lab command failed: '+json.dumps(receipt))
        return result.stdout, receipt

    def resolve(self):
        output, _ = self.command(['compose','-p','rca-lab','ps','--all','--no-trunc','--format','json'])
        self.ids = resolve_containers(compose_rows(output),self.topology.resources)
        return dict(self.ids)

    def state(self, resource):
        if resource not in self.ids:
            raise ValueError('unresolved resource')
        output, _ = self.command(['inspect',self.ids[resource]])
        rows = json.loads(output)
        if len(rows) != 1 or rows[0]['Id'] != self.ids[resource]:
            raise RuntimeError('container identity changed')
        labels = rows[0]['Config']['Labels']
        if labels.get('com.docker.compose.project') != 'rca-lab' or labels.get('com.docker.compose.service') != resource:
            raise RuntimeError('container registry mismatch')
        return {'resource':resource,'container_id':self.ids[resource],'state':rows[0]['State']['Status'],
                'sample_time':datetime.now(timezone.utc).isoformat()}

    def states(self):
        output, _ = self.command(['inspect',*self.ids.values()])
        rows = json.loads(output)
        by_id = {row['Id']:row for row in rows}
        if set(by_id) != set(self.ids.values()):
            raise RuntimeError('lab resource set changed')
        for resource, identifier in self.ids.items():
            labels = by_id[identifier]['Config']['Labels']
            if labels.get('com.docker.compose.project') != 'rca-lab' or labels.get('com.docker.compose.service') != resource:
                raise RuntimeError('container registry mismatch')
        return {r:by_id[i]['State']['Status'] for r,i in self.ids.items()}

    def inject_stop(self, resource):
        if resource not in ('catalogue-db','orders'):
            raise ValueError('unapproved stop fault')
        if self.state(resource)['state'] != 'running':
            raise RuntimeError('target is not initially running')
        _, receipt = self.command(['stop','--time','3',self.ids[resource]])
        return {**receipt,'resource':resource,'container_id':self.ids[resource],'mechanism':'container_stopped'}

    def recover(self, resource):
        if resource not in ('catalogue-db','orders'):
            raise ValueError('unapproved recovery')
        # Exact identity is rechecked before a recovery command.
        self.state(resource)
        _, receipt = self.command(['start',self.ids[resource]])
        return receipt

    def host_context(self):
        output,_=self.command(['info','--format','{{json .}}'])
        info=json.loads(output)
        context={key:info.get(key) for key in ('NCPU','MemTotal','OSType','Architecture','ServerVersion')}
        self.resolve()
        output,_=self.command(['stats','--no-stream','--format','{{json .}}',*self.ids.values()])
        if len(output.encode())>16384:
            raise RuntimeError('host statistics exceeded limit')
        context['container_load_samples']=[{key:row.get(key) for key in ('Name','CPUPerc','MemUsage','PIDs')}
            for row in (json.loads(line) for line in output.splitlines() if line.strip())]
        context['sample_time']=datetime.now(timezone.utc).isoformat()
        return context
