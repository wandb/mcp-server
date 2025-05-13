"""
Pytest configuration file.
"""

import os
import pytest
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Ensure WANDB_SILENT is set
os.environ["WANDB_SILENT"] = "true"

@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Set up the test environment."""
    # Set up environment variables
    os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"