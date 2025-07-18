#!/bin/bash

# Load environment variables (handle comments properly)
if [ -f config.env ]; then
    # Use source to load the file properly, handling comments and special characters
    set -a  # automatically export all variables
    source config.env
    set +a  # turn off automatic export
fi

# Ensure directories exist
mkdir -p temp_uploads temp_results

# Start the application
exec gunicorn pdf_processor_api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 