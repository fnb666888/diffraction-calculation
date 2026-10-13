$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed. Please install Python 3.11 or newer.' }
}
& $taskPython -c "import LightPipes, numpy, scipy, matplotlib, PIL"
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install --index-url https://pypi.org/simple -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $taskPython -m optics_sim gui --config configs/default.json
