$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$env:GOCACHE = Join-Path $taskRoot '.go-cache'
$env:GOOS = 'linux'
$env:GOARCH = 'amd64'
$env:CGO_ENABLED = '0'
$env:GO111MODULE = 'off'
& go build -trimpath -o (Join-Path $taskRoot 'lab/path-proxy/rca-path-proxy') (Join-Path $taskRoot 'lab/path-proxy/main.go')
if ($LASTEXITCODE -ne 0) { throw 'Proxy build failed' }
& docker build -t rca-path-proxy:v2 (Join-Path $taskRoot 'lab/path-proxy')
if ($LASTEXITCODE -ne 0) { throw 'Proxy image build failed' }
& docker image inspect rca-path-proxy:v2 --format '{{.Id}}'
