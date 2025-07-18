# PDF Processor API Deployment Guide

## Overview
This Flask application processes PDF files using Google Cloud Vision API and extracts data into Excel templates.

## Deployment Files Created

### 1. `requirements.txt`
- Added `gunicorn==21.2.0` for production WSGI server
- All other dependencies remain the same

### 2. `Procfile`
- Tells Render how to start the application
- Uses `bash start.sh` as the start command

### 3. `start.sh`
- Startup script that loads environment variables
- Creates necessary directories
- Starts the application with Gunicorn

### 4. `runtime.txt`
- Specifies Python version 3.8.18

### 5. `apt-packages`
- Lists system dependencies (poppler-utils for pdf2image)

### 6. `render.yaml`
- Render deployment configuration
- Sets up health checks and environment variables

## Key Changes Made

### Flask Application (`pdf_processor_api.py`)
1. **Port Configuration**: Now uses `PORT` environment variable
2. **Debug Mode**: Disabled by default, can be enabled via `DEBUG` environment variable
3. **Root Endpoint**: Added `/` endpoint for basic connectivity testing
4. **Health Check**: Fixed field mappings reference in health endpoint

## Environment Variables Required

Make sure to set these in your Render dashboard:

### Required
- `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_CLOUD_VISION_API_KEY`
- All field mapping coordinates (see `config.env` for examples)

### Optional
- `DEBUG`: Set to 'true' to enable debug mode
- `PORT`: Automatically set by Render

## Deployment Steps

1. **Push to Git**: Ensure all files are committed and pushed
2. **Connect to Render**: Link your repository to Render
3. **Set Environment Variables**: Add all required environment variables
4. **Deploy**: Render will automatically detect the Python app and deploy

## Health Check Endpoints

- `/` - Basic connectivity test
- `/api/health` - Detailed health check with service status

## Troubleshooting

### Common Issues

1. **Deployment Timeout**: 
   - Ensure all environment variables are set
   - Check that `start.sh` is executable
   - Verify `requirements.txt` is complete

2. **Import Errors**:
   - Make sure all packages are in `requirements.txt`
   - Check Python version compatibility

3. **Vision API Issues**:
   - Verify Google Cloud credentials are properly set
   - Check API key or service account permissions

### Testing Locally

Run the test script to verify configuration:
```bash
python3 test_deployment.py
```

### Logs

Check Render logs for detailed error information:
- Build logs show package installation issues
- Runtime logs show application startup and runtime errors

## Performance Notes

- Using 1 worker process to avoid memory issues
- 300-second timeout for long-running PDF processing
- Automatic cleanup of old files every hour 