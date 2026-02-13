# PowerShell script to sync CS336 repositories from GaoLeiA

$repos = @(
    "spring2025-lectures",
    "assignment1-basics",
    "assignment2-systems",
    "assignment3-scaling",
    "assignment4-data",
    "assignment5-alignment"
)

$baseUrl = "https://github.com/GaoLeiA"

Write-Host "Starting CS336 Repository Sync..." -ForegroundColor Cyan

foreach ($repo in $repos) {
    Write-Host "`nChecking $repo..." -ForegroundColor Yellow
    
    if (Test-Path $repo) {
        Write-Host "  Directory exists. Pulling latest changes..."
        Push-Location $repo
        try {
            git pull
        } catch {
            Write-Error "  Failed to pull $repo"
        }
        Pop-Location
    } else {
        Write-Host "  Directory not found. Cloning..."
        $url = "$baseUrl/$repo.git"
        try {
            git clone $url
        } catch {
            Write-Error "  Failed to clone $repo"
        }
    }
}

Write-Host "`nAll operations completed." -ForegroundColor Green
Read-Host "Press Enter to exit"
