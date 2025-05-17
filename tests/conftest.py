import pytest
import weave
import os
import logging

os.environ["WANDB_SILENT"] = "True"
weave_logger = logging.getLogger("weave")
weave_logger.setLevel(logging.ERROR)

# Configure logging early
logger = logging.getLogger(__name__)

# Ensure WANDB_SILENT is set if needed
os.environ["WANDB_SILENT"] = "true"

@pytest.fixture(scope="session", autouse=True)
def setup_weave(request):
    """Initializes Weave once per test session."""
    project_name = "wandb-applied-ai-team/wandb-mcp-server-test-suite-outputs"
    logger.info(f"Initializing Weave for project: {project_name}")
    try:
        # Attempt initialization
        # Check if already initialized (simple check, might need refinement based on weave internals)
        # Using weave.client() might be a more robust way to check, but requires knowing its state.
        # For now, let's assume init handles idempotency or rely on a simpler check if available.
        # Reverting to just calling init, assuming it handles being called multiple times gracefully
        # or that the session scope prevents redundant calls in separate processes.
        weave.init(project_name)
        logger.info(f"Weave initialized successfully for {project_name}")

    except Exception as e:
        # Log the error and fail the session setup if initialization fails critically
        logger.error(f"Fatal error during Weave initialization: {e}", exc_info=True)
        pytest.fail(f"Weave initialization failed: {e}", pytrace=False)

    # Optional: Teardown logic if needed
    # def fin():
    #     logger.info("Tearing down Weave session (if necessary)...")
    # request.addfinalizer(fin)
