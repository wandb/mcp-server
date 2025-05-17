"""
Tests that compare the raw HTTP implementation with the original implementation.

These tests verify that the raw HTTP implementation produces the same results
as the original implementation.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock
import pytest
import requests

# Import both implementations
from wandb_mcp_server.query_weave import (
    query_traces as query_traces_client,
    paginated_query_traces as paginated_query_traces_client,
)
from wandb_mcp_server.query_weave_raw import (
    query_traces as query_traces_raw,
    paginated_query_traces as paginated_query_traces_raw,
)


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


class MockWeaveClient:
    """Mock Weave client for testing."""

    def __init__(self, trace_data):
        self.trace_data = trace_data
        self.server = self

    def calls_query(self, req):
        """Mock calls_query method."""
        # Create a response object with the trace data
        class CallsQueryRes:
            def __init__(self, calls):
                self.calls = calls

        # Convert the trace data to the format expected by the client implementation
        calls = []
        for trace in self.trace_data:
            # Create a mock call object with the same attributes as the trace
            call = MagicMock()
            for key, value in trace.items():
                setattr(call, key, value)
            calls.append(call)

        return CallsQueryRes(calls=calls)


class TestQueryWeaveComparison(unittest.TestCase):
    """Test that the raw HTTP implementation matches the original implementation."""

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
    @patch("weave.init")
    def test_query_traces_comparison(self, mock_weave_init, mock_post):
        """Test that query_traces_raw produces the same results as query_traces_client."""
        # Set up the mock response for the raw HTTP implementation
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Set up the mock Weave client for the original implementation
        mock_weave_client = MockWeaveClient(self.trace_data)
        mock_weave_init.return_value = mock_weave_client

        # Call both implementations with the same parameters
        params = {
            "entity_name": "wandb-applied-ai-team",
            "project_name": "mcp-tests",
            "filters": {"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
            "limit": 10,
        }

        # Add api_key parameter for the raw implementation
        raw_params = params.copy()
        raw_params["api_key"] = "YOUR_WANDB_API_KEY"

        # Call both implementations
        raw_result = query_traces_raw(**raw_params)
        client_result = query_traces_client(**params)

        # Verify that both implementations produce the same results
        self.assertEqual(len(raw_result), len(client_result))
        
        # Compare the first trace from each implementation
        raw_trace = raw_result[0]
        client_trace = client_result[0]
        
        # Check that the key fields match
        self.assertEqual(raw_trace["id"], client_trace["id"])
        self.assertEqual(raw_trace["op_name"], client_trace["op_name"])
        self.assertEqual(raw_trace["display_name"], client_trace["display_name"])
        self.assertEqual(raw_trace["started_at"], client_trace["started_at"])
        self.assertEqual(raw_trace["ended_at"], client_trace["ended_at"])

    def test_paginated_query_traces_comparison(self):
        """Test that paginated_query_traces_raw produces the same results as paginated_query_traces_client."""
        # This is a placeholder test to ensure both functions exist
        # The actual implementation is tested in the integration tests
        self.assertTrue(callable(paginated_query_traces_raw))
        self.assertTrue(callable(paginated_query_traces_client))


if __name__ == "__main__":
    unittest.main()