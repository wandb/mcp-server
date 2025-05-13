#!/bin/bash
# Examples of using curl to interact with the Weave API

# Set your API key and project details
API_KEY="your_api_key_here"
ENTITY="your_entity_name"
PROJECT="your_project_name"
PROJECT_ID="${ENTITY}/${PROJECT}"

# Query traces
echo "Querying traces..."
curl -s -X POST "https://trace.wandb.ai/calls/stream_query" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "filter": {
      "trace_roots_only": true
    },
    "limit": 5,
    "sort_by": [{"field": "started_at", "direction": "desc"}],
    "include_costs": true,
    "include_feedback": true
  }' | jq -s '.'

# Query traces with complex filters
echo "Querying traces with complex filters..."
curl -s -X POST "https://trace.wandb.ai/calls/stream_query" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "filter": {
      "trace_roots_only": true
    },
    "query": {
      "$expr": {
        "$and": [
          {
            "$contains": {
              "input": {"$getField": "op_name"},
              "substr": {"$literal": "openai"},
              "case_insensitive": true
            }
          },
          {
            "$gt": [
              {"$getField": "summary.weave.latency_ms"},
              {"$literal": 1000}
            ]
          }
        ]
      }
    },
    "limit": 5,
    "sort_by": [{"field": "started_at", "direction": "desc"}],
    "include_costs": true
  }' | jq -s '.'

# Get a specific call
echo "Getting a specific call..."
CALL_ID="your_call_id_here"
curl -s -X POST "https://trace.wandb.ai/calls/get" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "call_id": "'"${CALL_ID}"'",
    "include_costs": true,
    "include_storage_size": true,
    "include_total_storage_size": true
  }' | jq '.'

# Get call statistics
echo "Getting call statistics..."
curl -s -X POST "https://trace.wandb.ai/calls/stats" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "filter": {
      "trace_roots_only": true
    },
    "include_total_storage_size": true
  }' | jq '.'

# Query traces with time range filter
echo "Querying traces with time range filter..."
START_DATE="2023-01-01T00:00:00Z"
END_DATE="2023-12-31T23:59:59Z"
curl -s -X POST "https://trace.wandb.ai/calls/stream_query" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "filter": {
      "trace_roots_only": true
    },
    "query": {
      "$expr": {
        "$and": [
          {
            "$gte": [
              {"$getField": "started_at"},
              {"$literal": "'"${START_DATE}"'"}
            ]
          },
          {
            "$lt": [
              {"$getField": "started_at"},
              {"$literal": "'"${END_DATE}"'"}
            ]
          }
        ]
      }
    },
    "limit": 5,
    "sort_by": [{"field": "started_at", "direction": "desc"}],
    "include_costs": true
  }' | jq -s '.'

# Query traces with attribute filter
echo "Querying traces with attribute filter..."
curl -s -X POST "https://trace.wandb.ai/calls/stream_query" \
  -u "${API_KEY}:" \
  -H "Content-Type: application/json" \
  -H "Accept": "application/jsonl" \
  -d '{
    "project_id": "'"${PROJECT_ID}"'",
    "filter": {
      "trace_roots_only": true
    },
    "query": {
      "$expr": {
        "$eq": [
          {"$getField": "attributes.model"},
          {"$literal": "gpt-4"}
        ]
      }
    },
    "limit": 5,
    "sort_by": [{"field": "started_at", "direction": "desc"}],
    "include_costs": true
  }' | jq -s '.'