import pytest
import time
import uuid
import logging
import os

# Minimal logger setup
logger = logging.getLogger("pytest_hook_test")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Comment out all Weave and other complex setup for this diagnostic phase
# os.environ["WEAVE_DISABLED"] = "true"
# ... other commented out original conftest code ...

def pytest_sessionfinish(session):
    invocation_id = str(uuid.uuid4())
    worker_id = "master"
    if hasattr(session.config, 'workerinput') and session.config.workerinput is not None: # Ensure workerinput is not None
        worker_id = session.config.workerinput.get('workerid', 'worker_unknown') # e.g., "gw0"
    
    # Create the log file regardless of whether it's master or worker for diagnostic purposes
    log_file_name = f"session_finish_minimal_log_pid_{os.getpid()}_{worker_id}_{invocation_id}.txt"
    with open(log_file_name, "w") as f:
        f.write(f"pytest_sessionfinish invoked (ID: {invocation_id}, ProcessID: {os.getpid()}, Worker: {worker_id}) at {time.time()}\n")
    logger.info(f"pytest_sessionfinish invoked (ID: {invocation_id}, ProcessID: {os.getpid()}, Worker: {worker_id})")

    if worker_id == "master":
        logger.info(f"Running main logic in pytest_sessionfinish because worker_id is 'master' (ID: {invocation_id})")
        #
        # >>>>>>>>>> HERE, you would re-insert your FULL pytest_sessionfinish logic <<<<<<<<<<
        # (i.e., the version that sets WEAVE_DISABLED=false, calls weave.init(),
        # finds JSONs, creates EvaluationLogger, logs predictions, and logs summary)
        # For now, to keep this test minimal, we'll just log that master would run.
        #
        logger.info(f"Master process (ID: {invocation_id}) would now perform Weave aggregation.")
    else:
        logger.info(f"Skipping main logic in pytest_sessionfinish because worker_id is '{worker_id}' (ID: {invocation_id})")
