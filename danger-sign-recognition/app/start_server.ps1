$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Serving danger-sign recognition frontend at http://localhost:8000"
Write-Host "Press Ctrl+C to stop."
python -m http.server 8000 -d $AppDir
