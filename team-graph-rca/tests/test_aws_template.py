import json
import pytest
from rca_investigation.backends.aws_template import EC2StatusTemplate
from rca_investigation.contracts import canonical


def template():
    return EC2StatusTemplate(('ap-south-1',),canonical({'catalogue-db':{'type':'ec2','instance_id':'i-0123456789abcdef0','region':'ap-south-1'}}))


def test_bounded_exact_registry_and_command():
    tool=template()
    argv=tool.argv('catalogue-db')
    tool.validate_argv('catalogue-db',argv)
    assert '--no-paginate' in argv and '--include-all-instances' in argv
    assert 'describe-instance-status' in tool.bash('catalogue-db')
    with pytest.raises(ValueError): tool.argv('$(malicious)')
    with pytest.raises(ValueError): tool.validate_argv('catalogue-db',argv+['--endpoint-url','http://evil'])


def test_missing_partial_or_wrong_identity_never_reports_stopped():
    tool=template()
    assert tool.normalize('catalogue-db','{"InstanceStatuses":[]}')['availability']=='NO_DATA'
    with pytest.raises(ValueError): tool.normalize('catalogue-db','{"InstanceStatuses":[],"NextToken":"more"}')
    with pytest.raises(ValueError): tool.normalize('catalogue-db',json.dumps({'InstanceStatuses':[{'InstanceId':'wrong'}]}))
    output={'InstanceStatuses':[{'InstanceId':'i-0123456789abcdef0','InstanceState':{'Name':'running'}}]}
    assert tool.normalize('catalogue-db',json.dumps(output))['response_code']=='RUNNING'
    with pytest.raises(ValueError): EC2StatusTemplate(('ap-south-1',),canonical({'x':{'type':'ec2','instance_id':'i-abc;rm','region':'ap-south-1'}}))
