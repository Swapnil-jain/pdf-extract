#!/usr/bin/env python3
"""
Update automation status in Supabase - Fixed Version
Does not fail the workflow on errors
"""

import os
import sys
from datetime import datetime

def update_automation_status():
    """Update the automation status in Supabase"""
    try:
        # Get environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        user_id = os.getenv('USER_ID')
        run_id = os.getenv('RUN_ID')
        job_status = os.getenv('JOB_STATUS', 'unknown')
        
        print(f"Updating status with job_status: {job_status}")
        
        if not all([supabase_url, supabase_key, user_id, run_id]):
            print("Missing required environment variables - skipping status update")
            return
        
        # Try to import and use Supabase
        try:
            from supabase import create_client
            
            # Initialize Supabase client
            supabase = create_client(supabase_url, supabase_key)
            
            # Determine status - GitHub Actions job.status can be 'success', 'failure', 'cancelled'
            if job_status.lower() in ['success', 'completed']:
                status = 'completed'
            elif job_status.lower() in ['failure', 'failed']:
                status = 'failed'
            elif job_status.lower() in ['cancelled', 'canceled']:
                status = 'cancelled'
            else:
                status = 'unknown'
            
            # Update automation status
            status_record = {
                'user_id': user_id,
                'run_id': run_id,
                'status': status,
                'completed_at': datetime.now().isoformat(),
                'github_run_id': run_id,
                'logs_available': True,
                'job_status': job_status  # Store original job status for debugging
            }
            
            # Insert or update status
            result = supabase.table('automation_status').upsert(status_record, on_conflict='run_id').execute()
            
            print(f"Status updated successfully: {status} for run {run_id}")
            print(f"Supabase response: {len(result.data) if result.data else 0} records affected")
            
        except ImportError:
            print("Supabase library not available - skipping status update")
        except Exception as supabase_error:
            print(f"Supabase error (non-critical): {supabase_error}")
        
    except Exception as e:
        print(f"Status update error (non-critical): {e}")
    
    # ALWAYS return success - status update failures should not fail the workflow
    print("Status update completed (success or skipped)")

if __name__ == "__main__":
    update_automation_status()
    # Always exit with success code
    sys.exit(0)

