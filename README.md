# MCP Server

A Model Context Protocol (MCP) server for querying [Weights & Biases Weave](https://weave-docs.wandb.ai/) traces. This server allows a MCP Client to:

- query W&B Weave traces
- write text and charts to W&B Reports

## Available tools

- **query_weave_traces_tool**: Queries Weave traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **count_weave_traces_tool**: Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.

- **create_wandb_report_tool**: Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

## Usage

Ensure you specify the W&B Entity and W&B Project to the LLM/MCP Client.

Example query for Claude Desktop:

```markdown
how many openai.chat traces in the wandb-applied-ai-team/mcp-tests weave project? plot the most recent 5 traces over time and save to a report
```

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Configuration

1. Create a `.env` file in the root directory with your Weights & Biases API key:
```
WANDB_API_KEY=your_api_key_here
```

## Running the Server

Run the server using:

```bash
uv run src/mcp_server/server.py
```

## Client Setup

### Claude Desktop

```json
    "mcpServers": {
        "weights_and_biases": {
        "command": "uv",
        "args": [
            "--directory",
            "/ABSOLUTE/PATH/TO/PROJECT",
            "run",
            "src/server.py"
        ]
        }
    }
```