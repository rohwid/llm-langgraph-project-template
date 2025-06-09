from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers.runs import router as runs_router
from src.api.routers.threads import router as threads_router
from src.api.routers.messages import router as messages_router

from loguru import logger

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)
app.include_router(threads_router)
app.include_router(messages_router)

# Health Check
@app.get("/")
async def read_root():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: A dictionary with a message indicating the API is up.
    """
    return {
        "message": "Up!"
    }

# Http error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handles HTTP exceptions raised by the application.
    
    Args:
        request (Request): The request that triggered the exception.
        exc (HTTPException): The HTTP exception to be handled.
    
    Returns:
        JSONResponse: A JSON response with the exception details.
    """
    logger.exception("HTTP exception.")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail
        }
    )

# Other error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handles general exceptions not caught by the HTTPException handler.
    
    Args:
        request (Request): The request that triggered the exception.
        exc (Exception): The exception to be handled.
    
    Returns:
        JSONResponse: A JSON response with a generic error message.
    """
    logger.exception("Unhandled exception.")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        }
    )