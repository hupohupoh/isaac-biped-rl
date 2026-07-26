# sync_to_hpc.ps1 — 全量同步到无法联网的 HPC
# 用法: .\sync_to_hpc.ps1

param(
    [string]$HPC = "2242211591@gpu03",
    [string]$RemotePath = "/home/2242211591/biped_demo"
)

$ErrorActionPreference = "Stop"
$LocalRoot = "F:\RobotProject\biped_demo"

# ── 1. 打包 git 追踪的所有源文件 ──────────────────────────────────
Write-Host "[1/3] 打包源代码..." -ForegroundColor Cyan
$archive = "$env:TEMP\biped_src.tar.gz"
Push-Location $LocalRoot
git archive --format=tar.gz -o $archive HEAD
Pop-Location
$sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 1)
Write-Host "      打包完成: $sizeMB MB"

# ── 2. 传输代码包 ──────────────────────────────────────────────────
Write-Host "[2/3] 传输源代码到 HPC..." -ForegroundColor Cyan
ssh $HPC "rm -rf $RemotePath.bak && mv $RemotePath $RemotePath.bak 2>/dev/null; mkdir -p $RemotePath"
scp $archive "${HPC}:$RemotePath/"
ssh $HPC "cd $RemotePath && tar xzf biped_src.tar.gz && rm biped_src.tar.gz"
Write-Host "      代码传输完成"

# ── 3. 传输 gitignore 的大文件（USD、SIF）──────────────────────────
Write-Host "[3/3] 传输大文件..." -ForegroundColor Cyan
$bigFiles = @(
    "source/biped_demo/biped_demo/tasks/manager_based/biped_demo/assets/robot/usd",
    "hpc/biped-train.sif",
    "hpc/biped-train.tar"
)
foreach ($item in $bigFiles) {
    $local = Join-Path $LocalRoot $item
    if (-not (Test-Path $local)) {
        Write-Warning "  跳过: $item (本地不存在)"
        continue
    }
    Write-Host "  $item" -ForegroundColor Gray
    ssh $HPC "mkdir -p $RemotePath/$(Split-Path $item -Parent)"
    if (Test-Path $local -PathType Container) {
        scp -r -q "$local/*" "${HPC}:$RemotePath/$item/"
    } else {
        scp -q $local "${HPC}:$RemotePath/$item"
    }
}

Remove-Item $archive -ErrorAction SilentlyContinue
Write-Host "`n全部完成！" -ForegroundColor Green
