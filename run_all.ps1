Write-Host "Starting SpeechOps System (API, MLflow, Prometheus, Grafana)..."

# Check if docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
}

# Run docker-compose with full profile (and build to ensure latest code)
docker-compose --profile full up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSystem started successfully!"
    Write-Host "Services available at:"
    Write-Host " - API:        http://localhost:8080"
    Write-Host " - MLflow:     http://localhost:5001"
    Write-Host " - Prometheus: http://localhost:9090"
    Write-Host " - Grafana:    http://localhost:3000"
} else {
    Write-Error "Failed to start services."
}
