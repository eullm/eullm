# EuLLM Chat launcher — starts the engine and opens the chat in the default browser.
#
# Behaviour:
#   1. Look for a GGUF model in %LOCALAPPDATA%\EuLLM\models\ (first .gguf wins).
#   2. If no model present: open a file picker so the user can choose one.
#   3. If still no model: print a friendly message and exit (no crash).
#   4. Start `eullm run <model>` in a child PowerShell window.
#   5. Wait until the API responds, then open http://localhost:11435/ in the browser.
#
# The engine keeps running until the user closes its window (Ctrl+C inside it
# or simply closing the terminal). This script exits as soon as the browser is open.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EullmExe  = Join-Path $ScriptDir "eullm.exe"
$ModelsDir = Join-Path $env:LOCALAPPDATA "EuLLM\models"
$UiPort    = 11435
$ApiPort   = 11434

if (-not (Test-Path $EullmExe)) {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not find eullm.exe at:`n$EullmExe`n`nReinstall EuLLM to fix this.",
        "EuLLM Chat", "OK", "Error") | Out-Null
    exit 1
}

# Ensure the models directory exists so the user has somewhere obvious to drop GGUFs.
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
}

# Pick a model: first .gguf in ModelsDir, else file-picker fallback.
$ModelPath = Get-ChildItem -Path $ModelsDir -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
             Sort-Object Length -Descending |
             Select-Object -First 1 -ExpandProperty FullName

if (-not $ModelPath) {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName PresentationFramework

    $msg = @"
No GGUF model found in:
  $ModelsDir

EuLLM is the runtime; you bring the model. Download any .gguf file
(e.g. Qwen3-8B-Q4_K_M from huggingface.co/Qwen/Qwen3-8B-GGUF) and either:

  - drop it into the models folder above, OR
  - click OK to pick one now from anywhere on disk.
"@
    [System.Windows.Forms.MessageBox]::Show(
        $msg, "EuLLM Chat — pick a model", "OKCancel", "Information") | Out-Null

    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title  = "Select a GGUF model"
    $dlg.Filter = "GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*"
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $ModelPath = $dlg.FileName
    }
}

if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    Write-Host "No model selected. Drop a .gguf into $ModelsDir and re-run." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    exit 0
}

Write-Host "Starting EuLLM with model:" -ForegroundColor Cyan
Write-Host "  $ModelPath"
Write-Host ""
Write-Host "API:     http://localhost:$ApiPort"
Write-Host "Chat UI: http://localhost:$UiPort/"
Write-Host ""

# Start the engine in a NEW PowerShell window so users can see logs and Ctrl+C it.
$EngineArgs = @(
    "-NoExit",
    "-NoProfile",
    "-Command",
    "& '$EullmExe' run '$ModelPath' --port $ApiPort --ui-port $UiPort"
)
Start-Process -FilePath "powershell.exe" -ArgumentList $EngineArgs -WindowStyle Normal | Out-Null

# Wait for the API to respond (up to 90s — model load on CPU can be slow).
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$UiPort/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        # Connection refused while loading — keep waiting.
    }
    Start-Sleep -Seconds 1
}

if ($ready) {
    Start-Process "http://localhost:$UiPort/"
} else {
    Write-Host "Engine did not respond within 90s. Check the engine window for errors." -ForegroundColor Yellow
}
