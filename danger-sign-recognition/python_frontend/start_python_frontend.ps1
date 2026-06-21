$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Port = 7860
$Checkpoint = Join-Path $ProjectRoot "model\artifacts_viewpoint\best_danger_sign_model.pt"

Get-CimInstance Win32_Process -Filter "name='python.exe' or name='python3.exe'" |
    Where-Object { $_.CommandLine -like "*python_frontend*server.py*" -or $_.CommandLine -like "*http.server*$Port*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "Loading trained model and serving Python frontend..."
Write-Host "Checkpoint: $Checkpoint"
Write-Host "URL: http://127.0.0.1:$Port/"
Write-Host "Press Ctrl+C to stop."

python (Join-Path $ScriptDir "server.py") --host 127.0.0.1 --port $Port --checkpoint $Checkpoint
