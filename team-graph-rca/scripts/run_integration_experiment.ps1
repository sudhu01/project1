$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $experimentRoot
$env:UV_CACHE_DIR = Join-Path (Split-Path -Parent $experimentRoot) '.uv-cache'
$pipelineRoot = Join-Path $experimentRoot 'runs/integration_v1/pipeline'
New-Item -ItemType Directory -Path $pipelineRoot -Force | Out-Null

function Invoke-ExperimentGate {
    param([string]$GateName, [string[]]$GateArguments)
    $consolePath = Join-Path $pipelineRoot ($GateName + '.console.json')
    & uv run --frozen rca-investigate @GateArguments > $consolePath
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $consolePath | Select-Object -Last 20
        throw "Experiment gate failed: $GateName"
    }
    Write-Output "$GateName passed."
}

# An already-running pilot owns the lab. Wait for its durable accepted slots.
$pilotStatePath = 'runs/integration_v1/captures/sockshop_pilot_v1/schedule.json'
$pilotLockPath = 'runs/integration_v1/captures/sockshop_pilot_v1/capture.lock'
while ($true) {
    $pilotState = if (Test-Path -LiteralPath $pilotStatePath) { Get-Content -LiteralPath $pilotStatePath -Raw | ConvertFrom-Json } else { $null }
    if ($pilotState -and @($pilotState.accepted.PSObject.Properties).Count -eq 30) { break }
    if (-not (Test-Path -LiteralPath $pilotLockPath)) { break }
    $leaseOwner = Get-Content -LiteralPath $pilotLockPath -Raw | ConvertFrom-Json
    if (-not (Get-Process -Id $leaseOwner.pid -ErrorAction SilentlyContinue)) {
        throw 'Pilot lease owner is absent. Inspect recovery before resuming.'
    }
    Start-Sleep -Seconds 10
}
Invoke-ExperimentGate 'capture-pilot' @('capture','--profile','sockshop_v1','--schedule','configs/capture_pilot.yaml','--output','runs/integration_v1/pipeline/pilot-capture.json')
if (-not (Test-Path -LiteralPath 'datasets/sockshop_pilot_v1/splits.json')) {
    Invoke-ExperimentGate 'split-pilot' @('split','--dataset','datasets/sockshop_pilot_v1','--schedule-state',$pilotStatePath,'--output','runs/integration_v1/pipeline/pilot-split.json')
}
Invoke-ExperimentGate 'fit-pilot' @('fit-model','--dataset','datasets/sockshop_pilot_v1','--model','runs/integration_v1/pipeline/pilot-model.json','--output','runs/integration_v1/pipeline/pilot-fit.json')
foreach ($baselineName in @('dependency-rule','dp')) {
    Invoke-ExperimentGate ('pilot-' + $baselineName) @('evaluate','--dataset','datasets/sockshop_pilot_v1','--model','runs/integration_v1/pipeline/pilot-model.json','--split','validation','--policy',$baselineName,'--output',('runs/integration_v1/pipeline/pilot-' + $baselineName + '.json'))
}
Write-Output 'Pilot complete. Thirty accepted physical incidents; scores are mechanics checks only.'

Invoke-ExperimentGate 'capture-evaluation' @('capture','--profile','sockshop_v1','--schedule','configs/capture_evaluation.yaml','--output','runs/integration_v1/pipeline/evaluation-capture.json')
if (-not (Test-Path -LiteralPath 'datasets/sockshop_v1/splits.json')) {
    Invoke-ExperimentGate 'split-evaluation' @('split','--dataset','datasets/sockshop_v1','--schedule-state','runs/integration_v1/captures/sockshop_v1/schedule.json')
}
Write-Output 'Phase 10 complete. Independent evaluation cohort and grouped splits frozen.'
Invoke-ExperimentGate 'fit-model' @('fit-model')
Write-Output 'Phase 12 complete. Evidence model fitted from training incidents only.'
foreach ($baselineName in @('dependency-rule','voi','dp')) {
    Invoke-ExperimentGate ('validation-' + $baselineName) @('evaluate','--split','validation','--policy',$baselineName,'--output',('runs/integration_v1/pipeline/validation-' + $baselineName + '.json'))
}
Write-Output 'Phase 13 complete. Rules and bounded planning validated on the same incidents.'
foreach ($learningMethod in @('expected-sarsa','two-step-sarsa')) {
    foreach ($trainingSeed in @(11,22,33,44,55)) {
        $latestPath = 'runs/integration_v1/training/' + $learningMethod + '/' + $trainingSeed + '/latest.json'
        $trainingArguments = @('train','--policy',$learningMethod,'--episodes','10000','--seeds',([string]$trainingSeed),'--output',('runs/integration_v1/pipeline/train-' + $learningMethod + '-' + $trainingSeed + '.json'))
        if (Test-Path -LiteralPath $latestPath) { $trainingArguments += '--resume' }
        Invoke-ExperimentGate ('train-' + $learningMethod + '-' + $trainingSeed) $trainingArguments
    }
}
Write-Output 'Phase 14 complete. Both tabular methods trained across five seeds.'
if (-not (Test-Path -LiteralPath 'experiments/local/final_v1.json')) {
    Invoke-ExperimentGate 'compare-validation' @('compare','--split','validation')
}
$testReportPath = 'runs/integration_v1/compare-test.json'
$testComplete = (Test-Path -LiteralPath $testReportPath) -and (Get-Content -LiteralPath $testReportPath -Raw | ConvertFrom-Json).passed
if (-not $testComplete) { Invoke-ExperimentGate 'compare-test' @('compare','--split','test') }
Write-Output 'Phase 15 complete. Frozen policies compared on reserved test incidents.'
Invoke-ExperimentGate 'challenges' @('challenge')
$frozenSplit = Get-Content -LiteralPath 'datasets/sockshop_v1/splits.json' -Raw | ConvertFrom-Json
$snapshotPath = 'datasets/sockshop_v1/' + $frozenSplit.splits.validation[0]
Invoke-ExperimentGate 'selected-investigation' @('investigate','--backend','replay','--incident',$snapshotPath,'--policy','selected')
Write-Output 'Selected frozen-window investigation saved. Application equality is checked separately.'
