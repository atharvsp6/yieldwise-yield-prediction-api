# ============================================================================
# YieldWise API - Main Entry Point for Deployment
# ============================================================================

"""
Main entry point for running the YieldWise API.
This file is used for deployment on platforms like Render, Heroku, etc.
"""

from yield_api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)