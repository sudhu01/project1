$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$env:GOCACHE = Join-Path $taskRoot '.go-cache'
$env:GOOS = 'linux'
$env:GOARCH = 'amd64'
$env:CGO_ENABLED = '0'
$env:GO111MODULE = 'off'
$helperDestination = Join-Path $taskRoot 'lab/adaptive-helper/rca-helper-lease'
& go build -trimpath -o $helperDestination (Join-Path $taskRoot 'lab/adaptive-helper/main.go')
if ($LASTEXITCODE -ne 0) { throw 'Helper build failed' }
$helperHash = (Get-FileHash -LiteralPath $helperDestination -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output $helperHash
