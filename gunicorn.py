import os
import json
import multiprocessing


workers = os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1)
loglevel = os.getenv("LOG_LEVEL", "info")
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "5000")
bind = f"{host}:{port}"
keepalive = os.getenv("KEEPALIVE", 24 * 60 * 60)  # 1 day
timeout = os.getenv("TIMEOUT", 60)  # 1 minute

worker_class = os.getenv("WORKER_CLASS")

accesslog = os.getenv("ACCESS_LOG", "-")
# this doesn't reformat yet for uvicorn
# https://github.com/encode/uvicorn/issues/389 - JW, 12/12/19
access_log_format = json.dumps(
    {
        "timestamp": "%(t)s",
        "host": "%(h)s",
        "method": "%(m)s",
        "url": "%(U)s",
        "status": "%(s)s",
        "protocol": "%(H)s",
        "response_length": "%(b)s",
        "referer": "%(f)s",
        "user_agent": "%(a)s",
    }
)
