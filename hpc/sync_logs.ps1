# Auto-sync HPC training logs every 10 minutes
$User = "2242211591"
$Ip   = "10.131.16.13"
$Key  = "$env:USERPROFILE\.ssh\id_hpc"
$RemoteDir = "/share/home/2242211591/logs/rsl_rl/biped_demo"
$LocalDir  = "F:\RobotProject\biped_demo\logs_hpc"
$Mins = 10

# mkdir
if (-not (Test-Path $LocalDir)) { New-Item -ItemType Directory $LocalDir -Force | Out-Null }

# Step 1: ensure key exists
if (-not (Test-Path $Key)) {
    Write-Host "Generating SSH key..." -ForegroundColor Yellow
    ssh-keygen -t ed25519 -f $Key -N '""' 2>&1 | Out-Null
    Write-Host "Key generated: $Key" -ForegroundColor Green
}

# Step 2: test connection (BatchMode = no password prompt, fail fast)
Write-Host "Testing SSH to ${User}@${Ip}..." -ForegroundColor Yellow
$test = ssh -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${User}@${Ip}" "echo ok" 2>&1

if ($test -ne "ok") {
    Write-Host "SSH key not yet on HPC. Uploading..." -ForegroundColor Yellow
    Write-Host "Enter your HPC password:" -ForegroundColor Cyan
    $pw = Read-Host -AsSecureString
    $cred = New-Object System.Management.Automation.PSCredential($User, $pw)
    $pass = $cred.GetNetworkCredential().Password

    # Upload via plink-style: use ssh with password
    $pubkey = Get-Content "$Key.pub" -Raw
    $cmd = "mkdir -p ~/.ssh && echo '$pubkey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    $result = ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${User}@${Ip}" $cmd 2>&1
    # Above will prompt for password but we can't automate that in pure ssh
    # Alternative: use scp to upload the key file
}

# Step 3: sync loop
$sshOpts = "-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

Write-Host "==============================================" -ForegroundColor Green
Write-Host "  HPC Log Sync  (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Remote : ${User}@${Ip}"
Write-Host "Local  : $LocalDir"
Write-Host ""

while ($true) {
    $t = Get-Date -Format "HH:mm:ss"
    Write-Host "[$t] Syncing..." -ForegroundColor Gray

    # Copy logs (BatchMode = no password, fail silent if key not set up)
    scp -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -r "${User}@${Ip}:$RemoteDir" "$LocalDir\" 2>&1 | Out-Null

    # Copy SLURM logs
    scp -i $Key -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${User}@${Ip}:/share/home/2242211591/train_*.log" "$LocalDir\" 2>&1 | Out-Null

    # Show latest checkpoint
    $latest = Get-ChildItem -Path $LocalDir -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        $n = $latest.BaseName -replace "model_", ""
        Write-Host "  Checkpoint: iter $n" -ForegroundColor Cyan
    } else {
        Write-Host "  No checkpoint yet" -ForegroundColor Yellow
    }

    $next = (Get-Date).AddMinutes($Mins)
    Write-Host "  Next: $(Get-Date -Format 'HH:mm:ss' -Date $next)"
    Write-Host ""
    Start-Sleep -Seconds ($Mins * 60)
}
