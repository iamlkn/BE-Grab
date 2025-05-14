from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time

from app.api.v1.endpoints import connections, eda, datasets, cleaning_jobs, automl_sessions, chatbots, ai_summary, chart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], # Change to specific origins
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Logging Middleware
@app.middleware('http')
async def dispatch(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time()-start_time
    logger.info(f'{request.method} {request.url.path} - {response.status_code} - {process_time:2f}s')
    return response

# Include routers
app.include_router(connections.router)
app.include_router(datasets.router)
app.include_router(eda.router)
app.include_router(cleaning_jobs.router)
app.include_router(automl_sessions.router)
app.include_router(chatbots.router)
app.include_router(ai_summary.router)
app.include_router(chart.router)

@app.get('/')
async def main():
    return {'message': 'hello world'}
