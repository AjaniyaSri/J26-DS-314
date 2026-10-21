# Run this file in PowerShell from any folder. It uses your existing project's .venv.
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$App = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path $Python)) {
    Write-Error "Project .venv Python not found at $Python"
    exit 1
}
Set-Location $Repo
& $Python -m streamlit run $App
