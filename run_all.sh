#!/bin/bash

echo "Starting SpeechOps System (API, MLflow, Prometheus, Grafana)..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop and try again." >&2
    exit 1
fi

# Run docker-compose with full profile (and build to ensure latest code)
docker compose --profile full up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "System started successfully!"
    echo "Services available at:"
    echo " - API:        http://localhost:8080"
    echo " - MLflow:     http://localhost:5001"
    echo " - Prometheus: http://localhost:9090"
    echo " - Grafana:    http://localhost:3000"
else
    echo "Error: Failed to start services." >&2
    exit 1
fi
