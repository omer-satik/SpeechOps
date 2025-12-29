#!/bin/bash

echo "Stopping SpeechOps System..."

# Stop and remove containers, networks
docker compose --profile full down

if [ $? -eq 0 ]; then
    echo "System stopped successfully."
else
    echo "Error: Failed to stop services." >&2
    exit 1
fi  
