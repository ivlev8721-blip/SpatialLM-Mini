param(
    [ValidateSet('test', 'train', 'generate', 'benchmark', 'experiment')]
    [string]$Action = 'test',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
    $env:PYTHONUTF8 = '1'
    if ($Action -eq 'test') {
        & $pythonPath -m unittest discover -s tests -v @ExtraArgs
    } else {
        & $pythonPath -m "spatiallm.$Action" @ExtraArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "SpatialLM command exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
