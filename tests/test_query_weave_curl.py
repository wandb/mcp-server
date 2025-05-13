"""
Tests that verify the raw HTTP implementation works with the curl example.

These tests use the curl example provided in the PR to test the raw HTTP implementation.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock
import pytest
import requests

# Import the raw HTTP implementation
from wandb_mcp_server.query_weave_raw import query_traces


class MockResponse:
    """Mock response object for requests."""

    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self._content = content
        self.headers = headers or {}
        self._lines = content.splitlines() if isinstance(content, bytes) else []
        self._line_index = 0

    def json(self):
        """Return the JSON content."""
        return json.loads(self.content)

    def iter_lines(self):
        """Iterate over the lines in the content."""
        for line in self._lines:
            yield line

    @property
    def text(self):
        """Return the text content."""
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8")
        return self.content


class TestQueryWeaveCurl(unittest.TestCase):
    """Test the raw HTTP implementation with the curl example."""

    def setUp(self):
        """Set up the test environment."""
        # Load the fixture data
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "trace_response.json")
        with open(fixture_path, "r") as f:
            self.trace_data = json.load(f)

        # Convert the trace data to bytes for the mock response
        self.trace_data_bytes = b"\n".join([json.dumps(trace).encode("utf-8") for trace in self.trace_data])

        # Set up environment variables
        os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
        os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

    @patch("requests.post")
    def test_query_traces_with_curl_example(self, mock_post):
        """Test that query_traces works with the curl example."""
        # Set up the mock response
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Call the function with the parameters from the curl example
        result = query_traces(
            entity_name="wandb-applied-ai-team",
            project_name="mcp-tests",
            filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
            limit=10,
            api_key="YOUR_WANDB_API_KEY",
        )

        # Verify the result
        self.assertEqual(len(result), len(self.trace_data))
        self.assertEqual(result[0]["id"], self.trace_data[0]["id"])
        self.assertEqual(result[0]["op_name"], self.trace_data[0]["op_name"])

        # Verify the request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify the request body matches the curl example
        request_body = json.loads(kwargs["data"])
        self.assertEqual(request_body["project_id"], "wandb-applied-ai-team/mcp-tests")
        self.assertEqual(request_body["filter"]["call_ids"], ["01958ab9-3c68-7c23-8ccd-c135c7037769"])
        self.assertEqual(request_body["limit"], 10)
        
        # Verify the authentication matches the curl example
        self.assertEqual(kwargs["auth"], ("YOUR_WANDB_API_KEY", ""))
        
        # Verify the headers match the curl example
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["headers"]["Accept"], "application/jsonl")


if __name__ == "__main__":
    unittest.main()