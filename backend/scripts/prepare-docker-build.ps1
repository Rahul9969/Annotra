# Verify required weights exist in backend/ before docker build / Render deploy.
$Backend = Split-Path $PSScriptRoot -Parent
$Required = @("best.pt")
$Recommended = @("mobile_sam.pt", "yolov8s-worldv2.pt")
$Optional = @("model_nano32.tflite", "model32.tflite")

$fail = $false
foreach ($name in $Required) {
    $p = Join-Path $Backend $name
    if (-not (Test-Path $p)) {
        Write-Host "MISSING (required): $name" -ForegroundColor Red
        $fail = $true
    } else {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ('OK (required): {0} ({1} MB)' -f $name, $mb) -ForegroundColor Green
    }
}

foreach ($name in $Recommended) {
    $p = Join-Path $Backend $name
    if (-not (Test-Path $p)) {
        Write-Host "MISSING (recommended): $name - World model will download on first container start" -ForegroundColor Yellow
    } else {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ('OK (recommended): {0} ({1} MB)' -f $name, $mb) -ForegroundColor Green
    }
}

foreach ($name in $Optional) {
    $p = Join-Path $Backend $name
    if (Test-Path $p) {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ('OK (optional): {0} ({1} MB)' -f $name, $mb) -ForegroundColor DarkGray
    }
}

if ($fail) {
    Write-Host ''
    Write-Host "Copy best.pt into: $Backend" -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'Ready to build:' -ForegroundColor Cyan
Write-Host "  cd $Backend"
Write-Host '  docker build -t annotra-api .'
