import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('method',['expected-sarsa','two-step-sarsa'])
def test_actual_cli_resume_matches_uninterrupted(tmp_path,method):
    def run(folder,*extra):
        result=subprocess.run([sys.executable,'-c','from rca_investigation.cli import main; raise SystemExit(main())',
            'train-fixture','--name','perfect','--episodes','500','--policy',method,'--run-dir',str(folder),
            '--output',str(folder/'report.json'),*extra],capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
    full,split=tmp_path/'full',tmp_path/'split'
    run(full)
    run(split,'--stop-after','137')
    checkpoint=json.loads((split/'latest.json').read_text())
    assert checkpoint['episode']==137
    run(split,'--resume')
    assert json.loads((full/'latest.json').read_text())==json.loads((split/'latest.json').read_text())
    assert json.loads((full/'best.json').read_text())==json.loads((split/'best.json').read_text())
    result=subprocess.run([sys.executable,'-c','from rca_investigation.cli import main; raise SystemExit(main())',
        'train-fixture','--name','perfect','--episodes','501','--policy',method,'--run-dir',str(split),
        '--resume','--output',str(split/'changed.json')],capture_output=True,text=True,timeout=20)
    assert result.returncode==1 and 'changed' in result.stdout
