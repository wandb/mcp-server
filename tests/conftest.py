import pytest
import time
import uuid
import logging 
import os

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# All Weave/other complex setup is assumed to be commented out for this diagnostic step

def pytest_sessionfinish(session):
    invocation_id = str(uuid.uuid4())
    
    worker_id = "master"
    # getattr is safer: it returns None if 'workerinput' doesn't exist (e.g., on master)
    workerinput = getattr(session.config, 'workerinput', None)
    if workerinput is not None:
        worker_id = workerinput.get('workerid', 'worker_unknown') # e.g., "gw0"
    
    log_file_name = f"session_finish_minimal_log_pid_{os.getpid()}_{worker_id}_{invocation_id}.txt"
    with open(log_file_name, "w") as f:
        f.write(f"pytest_sessionfinish invoked (ID: {invocation_id}, ProcessID: {os.getpid()}, Worker: {worker_id}) at {time.time()}\n")
    logger.info(f"pytest_sessionfinish invoked (ID: {invocation_id}, ProcessID: {os.getpid()}, Worker: {worker_id})")

    if worker_id == "master":
        logger.info(f"MASTER_LOGIC_RUN: Running main logic in pytest_sessionfinish (ID: {invocation_id})")
        # >>> Placeholder for where your full Weave aggregation logic would go <<<
        # For this test, we just log that the master branch was taken.
    else:
        logger.info(f"WORKER_LOGIC_SKIP: Skipping main logic in pytest_sessionfinish for worker '{worker_id}' (ID: {invocation_id})")
