$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
& docker build --build-arg 'BASE_IMAGE=weaveworksdemos/payment@sha256:5ab1c9877480a018d4dda10d6dfa382776e6bca9fc1c60bacbb80903fde8cfe0' -t rca-payment-helper:v2 (Join-Path $taskRoot 'lab/adaptive-helper')
if ($LASTEXITCODE -ne 0) { throw 'Payment helper image build failed' }
& docker build --build-arg 'BASE_IMAGE=weaveworksdemos/carts@sha256:434d2f5a6e0e8beef1f253fe96f45b8437a703125fc003434c5282ecf8969a4f' -t rca-carts-helper:v2 (Join-Path $taskRoot 'lab/adaptive-helper')
if ($LASTEXITCODE -ne 0) { throw 'Carts helper image build failed' }
& docker image inspect rca-payment-helper:v2 rca-carts-helper:v2 --format '{{.Id}}'
