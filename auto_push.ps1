param(
    [string]$CommitMessage = "Auto update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

# Initialize git if not already initialized
if (!(Test-Path ".git")) {
    Write-Host "Initializing Git repository..."
    git init
}

# Ensure git user configuration
if (!(git config user.name)) {
    git config user.name "liyakhath4620-maker"
}
if (!(git config user.email)) {
    git config user.email "liyakhath4620@gmail.com"
}

# Ensure remote URL exists
$remoteUrl = git remote get-url origin 2>$null
if (!$remoteUrl) {
    $remoteUrl = Read-Host "Please enter your Git repository URL (e.g., https://github.com/username/repo.git)"
    if ($remoteUrl) {
        git remote add origin $remoteUrl
    } else {
        Write-Error "Remote URL is required to push."
        exit 1
    }
}

# Get current branch
$branch = git branch --show-current
if (!$branch) {
    $branch = "master"
    git branch -M master
}

# Check for modified or untracked files
$status = git status --porcelain
if ($status) {
    Write-Host "Staging changes..."
    git add .
    Write-Host "Committing changes with message: '$CommitMessage'..."
    git commit -m $CommitMessage
} else {
    Write-Host "Working tree is clean, no new changes to commit."
}

# Check if there are unpushed commits
$unpushed = git log "origin/$branch..HEAD" --oneline 2>$null
if ($unpushed -or $status) {
    Write-Host "Pushing changes to remote origin ($branch)..."
    git push -u origin $branch
    Write-Host "Push completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Local branch is already up to date with remote origin ($branch)." -ForegroundColor Cyan
}
