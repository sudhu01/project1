"""Extract compact tables from existing frozen reports."""
import json
from pathlib import Path
from statistics import mean

def read(path):
    return json.loads(Path(path).read_text())

def compact(s):
    return {k: s[k] for k in ('accuracy', 'mean_return', 'mean_probes', 'mean_credits', 'decision_latency_ms')}

result = {}
for profile in ('v2a', 'v2b'):
    base = Path('runs/adaptive_' + profile)
    test, frozen = read(base/'compare-test.json'), read(base/'final.json')
    assert test['passed'] and test['frozen_hash'] == frozen['hash']
    table = {}
    for method in ('expected-sarsa', 'two-step-sarsa'):
        for mode in ('tabular', 'hybrid'):
            rows = [test['summaries'][f'{method}-{seed}-{mode}'] for seed in (11,22,33,44,55)]
            table[method+'-'+mode] = {k: mean(r[k] for r in rows) for k in compact(rows[0])}
            table[method+'-'+mode]['seed_accuracy_range'] = [min(r['accuracy'] for r in rows),max(r['accuracy'] for r in rows)]
    for name in ('dp','dp-depth3','voi',frozen['selected_fixed'],'dependency-rule','generalized-rule','random','stop-immediately','defer-immediately'):
        if name in test['summaries']:
            table[name] = compact(test['summaries'][name])
    random_rows = [test['summaries'][f'random-valid-{seed}'] for seed in (11,22,33,44,55)]
    table['random-valid'] = {k: mean(r[k] for r in random_rows) for k in compact(random_rows[0])}
    challenge = read(base/'challenge.json')
    # Challenge reports retain method summaries for each declared scenario.
    challenge_table = {}
    for scenario, report in challenge['reports'].items():
        selected_report = report[frozen['selected']]
        summary = selected_report.get('summary', selected_report.get('summaries', selected_report))
        challenge_table[scenario] = compact(summary)
    result[profile] = dict(selected=frozen['selected'],selected_fixed=frozen['selected_fixed'],table=table,challenge=challenge_table)

cost = Path('runs/sockshop_adaptive_v2b/costs/8894d4bbbec7')
test, frozen = read(cost/'compare-test.json'),read(cost/'final.json')
result['expensive_probes'] = {name: compact(summary) for name,summary in test['summaries'].items()
    if name.startswith(('expected-sarsa-', 'two-step-sarsa-')) and name.endswith('-tabular') or name in ('dp',frozen['selected_fixed'])}
Path('runs/adaptive_v2/pre-v2c-benchmark-data.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
