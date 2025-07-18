#!/bin/bash

# Load environment variables
if [ -f config.env ]; then
    export $(cat config.env | grep -v '^#' | xargs)
fi

# Ensure directories exist
mkdir -p temp_uploads temp_results

# Start the application
exec gunicorn pdf_processor_api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 