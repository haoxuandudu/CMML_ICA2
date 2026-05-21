param(
    [string]$Output = "multivi_project_cloud_upload.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputPath = Join-Path $root $Output

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

$required = @(
    "data/10x-Multiome-Pbmc10k-RNA.h5ad",
    "data/10x-Multiome-Pbmc10k-ATAC.h5ad",
    "vendor/cobolt-0.0.1.zip",
    "requirements.txt",
    "notebooks/01_multivi_benchmark.py"
)

foreach ($item in $required) {
    $path = Join-Path $root $item
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $item"
    }
}

$items = @(
    "README.md",
    "report_main.md",
    "supporting_materials.md",
    "high_score_checklist.md",
    "requirements.txt",
    "run_on_cloud_gpu.sh",
    "notebooks",
    "data",
    "vendor",
    "figures",
    "tables",
    "results"
)

$paths = $items | ForEach-Object { Join-Path $root $_ }
Compress-Archive -LiteralPath $paths -DestinationPath $outputPath -Force
Write-Host "Created $outputPath"
