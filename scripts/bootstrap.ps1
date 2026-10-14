param(
    [ValidateSet('3.12', '3.13')]
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv is required. See https://docs.astral.sh/uv/getting-started/installation/'
    }
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.uv-python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot ".venv-$PythonVersion"
    uv sync --locked --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw 'Dependency sync failed.' }
    uv run --locked --python $PythonVersion python -I -m signai
    if ($LASTEXITCODE -ne 0) { throw 'Package import failed.' }
    uv run --locked --python $PythonVersion pytest
    if ($LASTEXITCODE -ne 0) { throw 'Smoke tests failed.' }
} finally {
    Pop-Location
}
