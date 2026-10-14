"""Summarize verified saved results without fitting or running policies."""
import json
from pathlib import Path

run = Path('runs/adaptive_v2c_memory_separation_r1')
read = lambda p: json.loads(Path(p).read_text())
evidence = read(run / 'evidence-sweep-verification.json')
cost = read(run / 'cost-sweep-verification.json')
main = read(run / 'main-comparison-verification.json')
challenge = read(run / 'main-challenge-verification.json')
metrics = read(run / 'method-report-metrics.json')
test = read(run / 'compare-test.json')
assert all(x['passed'] for x in (evidence, cost, main, challenge))
assert read(run / 'phase-progress.json')['passed']
assert len(test['frozen_method_mixture_sweeps']) > 0

lines = ['# Revised v2c benchmark and findings', '', '14 October 2026, IST', '',
 'The revised v2c experiment is complete. All 240 physical captures, main training and benchmarks, six main synthetic challenges, eleven evidence variants and five cost variants have finished. Saved reports passed verification. The original v2c failure remains preserved.', '',
 'VOI and depth-two DP achieved the best main-test results: 93.3% accuracy, score 0.72 and 2.3 probes. VOI reached the same result with much less decision computation. No SARSA method passed the declared promotion requirements in the main comparison or either revised v2c sweep.', '',
 '## What was measured', '',
 'The six classes were catalogue database stopped, orders stopped, payment CPU contention, orders-to-payment delay, carts memory pressure and healthy operation. The cohort has 120 training, 60 validation and 60 reserved-test incidents, grouped into 20, 10 and 10 independent sessions. The test contains ten cases per class.', '',
 'Physical capture ran a fault or healthy control in Docker and saved its observations. Replay reused those saved observations. Diagnostic fitting learned the relationship between evidence and causes. Policy training learned which evidence to buy next. The revision and sweeps required no additional physical captures.', '',
 'Both learners used seeds 11, 22, 33, 44 and 55, with 10,000 replay episodes per seed. The main revision plus sixteen sweep variants completed 170 training runs, or 1.7 million replay episodes. Across v2a, v2b and revised v2c, the totals are 600 physical captures and 450 cohort training runs, or 4.5 million replay episodes. Earlier synthetic fixture training is separate.', '',
 '## Main algorithm comparison', '',
 'Learner rows average all five seeds. Higher accuracy and score are better. Lower probes and computation are better when diagnosis quality is retained. Computation measures policy choice in replay, including terminal choices. It excludes Docker waiting and training time.', '',
 '| Method | Accuracy | Score | Average probes | Average credits | Computation per choice, ms |',
 '|---|---:|---:|---:|---:|---:|']
for name, row in metrics.items():
    lines.append(f"| {name} | {row['accuracy']:.1%} | {row['mean_return']:.4f} | {row['mean_probes']:.3f} | {row['mean_credits']:.3f} | {row['decision_latency_ms']:.3f} |")
lines += ['', 'The score rewards a correct diagnosis with 1, a wrong diagnosis with 0 and deferring with 0.25, then subtracts 0.05 times purchased credits in the main experiment. Deferring is not counted as a correct diagnosis. This explains why score can exceed accuracy in some reduced-evidence settings.', '',
 'The planners correctly diagnosed all fifty fault cases but only six of ten healthy cases. Their four mistakes were false fault diagnoses. A better memory boundary fixed the original validation failure, but it did not make limited-budget diagnosis perfect.', '',
 'Pure SARSA was fast but less accurate and used more probes. The hybrid can call a planner in states with few training visits. It improved average accuracy, but its fallback computation made it slower and its results still varied between seeds. The saved results support VOI for this cohort, not a claim that planning always beats RL.', '',
 '## Variation between training seeds', '',
 '| Learner form | Accuracy across five seeds | Score across five seeds |', '|---|---:|---:|']
for method in ('expected-sarsa', 'two-step-sarsa'):
    for form in ('tabular', 'hybrid'):
        rows = [test['summaries'][f'{method}-{s}-{form}'] for s in (11,22,33,44,55)]
        lines.append(f"| {method}, {form} | {min(r['accuracy'] for r in rows):.1%} to {max(r['accuracy'] for r in rows):.1%} | {min(r['mean_return'] for r in rows):.4f} to {max(r['mean_return'] for r in rows):.4f} |")
lines += ['', '## Evidence sweep', '',
 'Each subset kept all six causes and the same raw cohort. Each had a separate fitted model and trained checkpoints. The table shows the method selected on validation and then evaluated on reserved test.', '',
 '| Evidence available | Test accuracy | Score | Average probes | Diagnostic gate |', '|---|---:|---:|---:|---|']
for row in evidence['results']:
    lines.append(f"| {row['evidence']} | {row['test_accuracy']:.1%} | {row['test_return']:.4f} | {row['test_probes']:.3f} | {'Passed' if row['diagnostic_gate_passed'] else 'Failed'} |")
lines += ['', 'Only full evidence passed the 90% diagnostic validation gate. Removing memory or status reduced the selected test accuracy to 76.7%. Logs and request errors did not improve every subset. A tool can be useful for one fault while adding little information to another.', '',
 'Reduced subsets were predeclared comparative experiments. Their failed diagnostic gates remain recorded and block promotion. Finishing their runs does not mean they are suitable for deployment.', '',
 '## Cost sweep', '',
 'Cost weight controls how strongly the score penalizes buying evidence. These variants retrained learners for each declared weight. The score values use different objectives, so they should not be treated as one common ranking.', '',
 '| Cost weight | Selected method | Accuracy | Score | Average probes |', '|---|---|---:|---:|---:|']
for row in cost['results']:
    lines.append(f"| {row['lambda_cost']:.3f} | {row['selected']} | {row['test_accuracy']:.1%} | {row['test_return']:.4f} | {row['test_probes']:.3f} |")
lines += ['', 'At weight 0.1, VOI bought fewer probes and accuracy fell to 85%. At 0.2, the selected VOI policy deferred immediately on every case: zero probes, zero diagnosis accuracy and score 0.25. This is the declared reward encouraging abstention when evidence is expensive. It is not successful diagnosis.', '',
 '## Synthetic challenge checks', '',
 'These checks altered saved responses or public prices. They are not new physical outage experiments. The table uses the main validation-selected VOI policy.', '',
 '| Challenge | Accuracy | Score | Average probes |', '|---|---:|---:|---:|']
for name, row in challenge['results'].items():
    lines.append(f"| {name.replace('_', ' ')} | {row['accuracy']:.1%} | {row['mean_return']:.4f} | {row['mean_probes']:.3f} |")
lines += ['', 'Losing logs preserved the main result. Losing request metrics or container stats reduced accuracy. Losing all telemetry reduced accuracy to 33.3%. These results show which evidence the policy depends on; they do not establish resilience to arbitrary real outages.', '',
 '## Memory correction and training method', '',
 'The original frozen encoder set the carts pressure boundary at 57.734% of the memory limit. Normal validation use drifted slightly above that boundary, so healthy and delay cases were encoded as pressure. Full-evidence validation reached only 66.7%, and its 90% gate stopped training.', '',
 'The separate authorized revision fitted a declared rule using training data only: the midpoint between the largest healthy window median and the smallest verified pressure window median. The resulting threshold was 70.596%, between about 57.734% normal use and 83.457% injected pressure. Other encoder fields and accepted captures stayed unchanged.', '',
 'Revised validation reached 100%. Because the failure had already been investigated on validation, this is development validation, not independent proof of perfect diagnosis. Policies and settings were frozen before the revised reserved comparison. The original failed model and encoder remain separate.', '',
 'Training sampled saved training incidents with replacement. Alpha was 0.1, gamma was 1, and exploration decreased from 0.2 to 0.05 over 8,000 episodes. Validation every 500 episodes selected the checkpoint by score, with lower credits breaking ties. Evaluation used no exploration and made no Q updates. The model, encoder, source and checkpoint identities were frozen for comparison.', '',
 '## Promotion, crossover and limits', '',
 'The saved comparisons include frozen-method easy/hard mixture sweeps against the selected fixed baseline. They reweight recorded outcomes without retraining or changing priors. They describe this cohort, not a universal fault-count threshold for RL.', '',
 'Promotion required at least 90% accuracy, a return advantage of at least 0.02 over the selected fixed baseline, a positive paired session uncertainty bound and a passed diagnostic gate. No revised v2c learner passed. Earlier v2b promotions remain specific to their declared evidence subset and cost setting.', '',
 'The test has ten independent sessions. Perfect performance on a subset or a validation set does not establish general performance across deployments. Memory here means retained resource pressure; the captures did not establish a memory leak or customer degradation. Source-outage challenges are synthetic.', '',
 'The implementation and physical runs are complete, while successful RL promotion remains an experimental result to earn. The measurements support using VOI as the current main policy and retaining SARSA as a fast but less reliable alternative on this expanded cohort.', '',
 '## Saved evidence', '',
 '- `runs/adaptive_v2c_memory_separation_r1/main-comparison-verification.json`',
 '- `runs/adaptive_v2c_memory_separation_r1/main-challenge-verification.json`',
 '- `runs/adaptive_v2c_memory_separation_r1/evidence-sweep-verification.json`',
 '- `runs/adaptive_v2c_memory_separation_r1/cost-sweep-verification.json`',
 '- `runs/adaptive_v2c_memory_separation_r1/revision-declaration.json` and `memory-boundary-fit.json`',
 '- `docs/algorithm_working_and_training_report_2026-10-11.md` for algorithm theory and implementation details.', '']
destination = Path('docs/revised_v2c_benchmark_report_2026-10-14.md')
destination.write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'report': str(destination), 'mixture_methods_verified': len(test['frozen_method_mixture_sweeps']), 'words': len(' '.join(lines).split())}))
