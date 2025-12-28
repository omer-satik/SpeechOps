Write-Host "Stopping SpeechOps System..."

# Stop and remove containers, networks
docker-compose --profile full down

if ($LASTEXITCODE -eq 0) {
    Write-Host "System stopped successfully."
} else {
    Write-Error "Failed to stop services."
}
