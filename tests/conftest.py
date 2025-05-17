import uuid
import time

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

def pytest_sessionfinish(session):
    invocation_id = str(uuid.uuid4())
    with open(f"session_finish_log_{invocation_id}.txt", "w") as f:
        f.write(f"pytest_sessionfinish invoked (ID: {invocation_id}) at {time.time()}\n")
    logger.info(f"Minimal pytest_sessionfinish invoked (ID: {invocation_id})")
    # Comment out ALL Weave related logic, file finding, etc. for this test.
