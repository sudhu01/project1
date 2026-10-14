"""Preparation-only EC2 status template. No account or deployment is assumed."""
from dataclasses import dataclass
import json
import re
import shlex
from ..contracts import canonical


@dataclass(frozen=True)
class EC2StatusTemplate:
    allowed_regions: tuple
    registry_json: str
    timeout_seconds: int = 10
    max_bytes: int = 32768
    credits: int = 1

    def __post_init__(self):
        if type(self.allowed_regions) is not tuple or not self.allowed_regions or len(set(self.allowed_regions))!=len(self.allowed_regions) or any(type(r) is not str or not re.fullmatch(r'[a-z]{2}(?:-gov)?-[a-z]+-\d',r) for r in self.allowed_regions):
            raise ValueError('explicit allowed regions required')
        registry=json.loads(self.registry_json)
        if canonical(registry)!=self.registry_json or type(registry) is not dict or not registry:
            raise ValueError('canonical resource registry required')
        for resource,row in registry.items():
            if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_./-]{0,127}',resource) or type(row) is not dict or set(row)!={'type','instance_id','region'} or any(type(v) is not str for v in row.values()):
                raise ValueError('invalid resource registry entry')
            if row['type']!='ec2' or not re.fullmatch(r'i-(?:[a-f0-9]{8}|[a-f0-9]{17})',row['instance_id']) or row['region'] not in self.allowed_regions:
                raise ValueError('unsupported resource type, identifier or region')
        if any(type(v) is not int for v in (self.timeout_seconds,self.max_bytes,self.credits)) or not 1<=self.timeout_seconds<=15 or not 1024<=self.max_bytes<=32768 or not 1<=self.credits<=16:
            raise ValueError('invalid execution bounds')

    def argv(self, resource):
        registry=json.loads(self.registry_json)
        if resource not in registry: raise ValueError('unregistered resource')
        row=registry[resource]
        return ['aws','ec2','describe-instance-status','--instance-ids',row['instance_id'],
                '--region',row['region'],'--include-all-instances','--no-paginate','--output','json',
                '--no-cli-pager','--no-cli-auto-prompt','--cli-connect-timeout','3','--cli-read-timeout','5']

    def bash(self, resource):
        argv=self.argv(resource)
        command=shlex.join(argv)
        if shlex.split(command)!=argv: raise ValueError('command template round trip failed')
        return command

    def validate_argv(self, resource, argv):
        if argv!=self.argv(resource): raise ValueError('unapproved operation or parameters')

    def normalize(self, resource, output):
        if len(output.encode())>self.max_bytes: raise ValueError('output limit exceeded')
        payload=json.loads(output)
        if payload.get('NextToken'): raise ValueError('incomplete page; hidden pagination prohibited')
        rows=payload.get('InstanceStatuses')
        if type(rows) is not list: raise ValueError('invalid status payload')
        if not rows: return {'availability':'NO_DATA','response_code':'UNAVAILABLE','units':'instance_state'}
        identity=json.loads(self.registry_json)[resource]['instance_id']
        if len(rows)!=1 or rows[0].get('InstanceId')!=identity: raise ValueError('resource identity mismatch')
        state=rows[0].get('InstanceState',{}).get('Name')
        code='RUNNING' if state=='running' else 'STOPPED' if state=='stopped' else 'OTHER_STATE'
        if state is None: return {'availability':'NO_DATA','response_code':'UNAVAILABLE','units':'instance_state'}
        return {'availability':'OK','response_code':code,'units':'instance_state','state':state,
                'system_status':rows[0].get('SystemStatus',{}).get('Status'),
                'instance_status':rows[0].get('InstanceStatus',{}).get('Status')}
