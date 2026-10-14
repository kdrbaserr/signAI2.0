param([ValidateSet('cuda', 'cpu')][string]$Device = 'cuda')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.uv-python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv-3.12'
    uv sync --locked --extra probe --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Probe dependency sync failed.' }
    $probePython = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
    & $probePython scripts/prepare_moment_checkpoint.py
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint verification failed.' }
    & $probePython -m signai.smoke_train --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Model compatibility probe failed.' }
} finally {
    Pop-Location
}
