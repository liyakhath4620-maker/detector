param(
    [string]$CommitMessage = "Auto update"
)

# Initialize git if not already initialized
if (!(Test-Path ".git")) {
    Write-Host "Initializing Git repository..."
    git init
}

# Set the git user email
Write-Host "Configuring git user email..."
git config user.email "liyakhath4620@gmail.com"

# Ask for the remote URL if it's not set
$remoteExists = git remote get-url origin 2>$null
if (!$remoteExists) {
    $remoteUrl = Read-Host "Please enter your Git repository URL (e.g., https://github.com/username/repo.git)"
    if ($remoteUrl) {
        git remote add origin $remoteUrl
    } else {
        Write-Host "Remote URL is required to push."
        exit
    }
}

# Add, commit, and push
git add .
git commit -m $CommitMessage

# Ensure we are on a branch, default to main
$branch = git branch --show-current
if (!$branch) {
    $branch = "main"
    git branch -M main
}

Write-Host "Pushing to remote origin ($branch)..."
git push -u origin $branch
Write-Host "Done!"
