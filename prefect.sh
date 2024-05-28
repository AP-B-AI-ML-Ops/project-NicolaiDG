#!/bin/bash

# Set SQLite timeout to 30 seconds (30000 milliseconds)
export SQLITE_TIMEOUT=30000

# Initialize Prefect with the local recipe
echo "Initializing Prefect with local recipe..."
prefect init --recipe local
if [ $? -ne 0 ]; then
    echo "Failed to initialize Prefect."
    exit 1
fi

# Create a work pool named mlops
echo "Creating work pool named mlops..."
prefect work-pool create mlops --type process --set-as-default
if [ $? -ne 0 ]; then
    echo "Failed to create work pool."
    exit 1
fi

# Deploy the flow from main.py
echo "Deploying the flow from main.py..."
prefect --no-prompt deploy main.py:main_flow -n 'MLops' -p mlops
if [ $? -ne 0 ]; then
    echo "Failed to deploy the flow."
    exit 1
fi

# Start the Prefect server
echo "Starting Prefect server..."
prefect server start &
server_pid=$!
if [ $? -ne 0 ]; then
    echo "Failed to start Prefect server."
    exit 1
fi

# Start a worker for mlops
echo "Starting worker for mlops..."
prefect worker start -t process -p mlops
if [ $? -ne 0 ]; then
    echo "Failed to start worker."
    exit 1
fi

# Wait for the server to finish
wait $server_pid

echo "finished"
