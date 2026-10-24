# Recreate the tested Windows x64 / Python 3.12 environment from one requirements file.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$setupDirectory = Join-Path $projectRoot 'artifacts\setup'

function Invoke-ProjectPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $projectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Project Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
    }
    Invoke-ProjectPython -Arguments @('-c', 'import platform, sys; assert sys.platform == ''win32'' and sys.version_info[:2] == (3, 12) and platform.machine().upper() == ''AMD64'', ''Use Windows x64 and Python 3.12 for this tested environment.''')
    New-Item -ItemType Directory -Path $setupDirectory -Force | Out-Null
    & $projectPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Invoke-ProjectPython -Arguments @('-m', 'ensurepip', '--upgrade')
    }
    # requirements.txt also pins pip==25.2; no separate package install is needed.
    Invoke-ProjectPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements.txt')
    Invoke-ProjectPython -Arguments @('-m', 'ipykernel', 'install', '--sys-prefix', '--name', 'sc4002', '--display-name', 'SC4002 (.venv)')
    $env:PYTHONHASHSEED = '42'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    Invoke-ProjectPython -Arguments @('scripts/check_environment.py', '--device', 'cuda')
    Invoke-ProjectPython -Arguments @('-m', 'pip', 'check')
    # These offline checks also run in a fresh clone without local training artifacts.
    Invoke-ProjectPython -Arguments @('-m', 'unittest', 'tests.test_acquisition', 'tests.test_runtime', 'tests.test_metrics', 'tests.test_diagnostics', 'tests.test_notebook_build', '-v')
}
finally {
    Pop-Location
}
