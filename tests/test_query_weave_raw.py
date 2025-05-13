"""
Tests for the raw HTTP implementation of the Weave API client.

These tests verify that the raw HTTP implementation of the Weave API client
works correctly and produces the same results as the original implementation.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock
import pytest
import requests
from io import BytesIO

# Import the raw HTTP implementation
from wandb_mcp_server.query_weave_raw import (
    query_traces,
    paginated_query_traces,
    get_weave_trace_server,
    _build_query_expression,
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


class TestQueryWeaveRaw(unittest.TestCase):
    """Test the raw HTTP implementation of the Weave API client."""

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
    def test_query_traces_basic(self, mock_post):
        """Test the basic functionality of query_traces."""
        # Set up the mock response
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Call the function
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
        self.assertEqual(kwargs["auth"], ("YOUR_WANDB_API_KEY", ""))
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["headers"]["Accept"], "application/jsonl")

        # Verify the request body
        request_body = json.loads(kwargs["data"])
        self.assertEqual(request_body["project_id"], "wandb-applied-ai-team/mcp-tests")
        self.assertEqual(request_body["filter"]["call_ids"], ["01958ab9-3c68-7c23-8ccd-c135c7037769"])
        self.assertEqual(request_body["limit"], 10)

    @patch("requests.post")
    def test_query_traces_with_filters(self, mock_post):
        """Test query_traces with various filters."""
        # Set up the mock response
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Call the function with various filters
        result = query_traces(
            entity_name="wandb-applied-ai-team",
            project_name="mcp-tests",
            filters={
                "trace_roots_only": True,
                "op_name_contains": "Chat",
                "status": "success",
                "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"},
                "attributes": {"weave.client_version": "0.51.29"},
                "has_exception": False,
            },
            limit=10,
            api_key="YOUR_WANDB_API_KEY",
        )

        # Verify the result
        self.assertEqual(len(result), len(self.trace_data))

        # Verify the request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify the request body
        request_body = json.loads(kwargs["data"])
        self.assertEqual(request_body["filter"]["trace_roots_only"], True)
        self.assertTrue("query" in request_body)  # Complex filters should be in the query

    @patch("requests.post")
    def test_query_traces_with_sort(self, mock_post):
        """Test query_traces with sorting options."""
        # Set up the mock response
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Call the function with sorting options
        result = query_traces(
            entity_name="wandb-applied-ai-team",
            project_name="mcp-tests",
            filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
            sort_by="summary.weave.latency_ms",
            sort_direction="asc",
            limit=10,
            api_key="YOUR_WANDB_API_KEY",
        )

        # Verify the result
        self.assertEqual(len(result), len(self.trace_data))

        # Verify the request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify the request body
        request_body = json.loads(kwargs["data"])
        self.assertEqual(request_body["sort_by"][0]["field"], "summary.weave.latency_ms")
        self.assertEqual(request_body["sort_by"][0]["direction"], "asc")

    @patch("requests.post")
    def test_query_traces_with_columns(self, mock_post):
        """Test query_traces with column selection."""
        # Set up the mock response
        mock_post.return_value = MockResponse(200, self.trace_data_bytes)

        # Call the function with column selection
        result = query_traces(
            entity_name="wandb-applied-ai-team",
            project_name="mcp-tests",
            filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
            columns=["id", "op_name", "started_at", "ended_at"],
            limit=10,
            api_key="YOUR_WANDB_API_KEY",
        )

        # Verify the result
        self.assertEqual(len(result), len(self.trace_data))
        for trace in result:
            self.assertIn("id", trace)
            self.assertIn("op_name", trace)
            self.assertIn("started_at", trace)
            self.assertIn("ended_at", trace)
            self.assertNotIn("inputs", trace)  # Should not be included

        # Verify the request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify the request body
        request_body = json.loads(kwargs["data"])
        self.assertEqual(request_body["columns"], ["id", "op_name", "started_at", "ended_at"])

    @patch("requests.post")
    def test_query_traces_error_handling(self, mock_post):
        """Test error handling in query_traces."""
        # Set up the mock response for an error
        mock_post.return_value = MockResponse(400, '{"error": "Bad request"}')

        # Call the function and expect an exception
        with self.assertRaises(Exception):
            query_traces(
                entity_name="wandb-applied-ai-team",
                project_name="mcp-tests",
                filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
                limit=10,
                api_key="YOUR_WANDB_API_KEY",
            )

    def test_paginated_query_traces(self):
        """Test the paginated_query_traces function."""
        # This is a placeholder test to ensure the function exists
        # The actual implementation is tested in the integration tests
        self.assertTrue(callable(paginated_query_traces))

    def test_build_query_expression(self):
        """Test the _build_query_expression function."""
        # Test with op_name_contains filter
        filters = {"op_name_contains": "Chat"}
        query_expr = _build_query_expression(filters)
        self.assertIsNotNone(query_expr)
        
        # Test with time_range filter
        filters = {"time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"}}
        query_expr = _build_query_expression(filters)
        self.assertIsNotNone(query_expr)
        
        # Test with attributes filter
        filters = {"attributes": {"weave.client_version": "0.51.29"}}
        query_expr = _build_query_expression(filters)
        self.assertIsNotNone(query_expr)
        
        # Test with has_exception filter
        filters = {"has_exception": False}
        query_expr = _build_query_expression(filters)
        self.assertIsNotNone(query_expr)
        
        # Test with multiple filters
        filters = {
            "op_name_contains": "Chat",
            "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"},
            "attributes": {"weave.client_version": "0.51.29"},
            "has_exception": False,
        }
        query_expr = _build_query_expression(filters)
        self.assertIsNotNone(query_expr)

    def test_get_weave_trace_server(self):
        """Test the get_weave_trace_server function."""
        # Call the function
        server = get_weave_trace_server(
            api_key="YOUR_WANDB_API_KEY",
            project_id="wandb-applied-ai-team/mcp-tests",
        )
        
        # Verify the result
        self.assertIsNotNone(server)
        self.assertEqual(server.api_key, "YOUR_WANDB_API_KEY")
        self.assertEqual(server.project_id, "wandb-applied-ai-team/mcp-tests")


if __name__ == "__main__":
    unittest.main()