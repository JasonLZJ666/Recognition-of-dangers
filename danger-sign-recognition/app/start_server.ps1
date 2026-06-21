$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8000

Get-CimInstance Win32_Process -Filter "name='python.exe' or name='python3.exe'" |
    Where-Object { $_.CommandLine -like "*http.server*$Port*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "Serving danger-sign recognition frontend at http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop."
python -m http.server $Port -d $AppDir --bind 127.0.0.1
