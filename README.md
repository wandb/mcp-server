# Weights & Biases MCP Server

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
            "src/mcp_server/server.py"
        ]
        }
    }
```

## Troubleshooting

### Error: spawn uv ENOENT

If you encounter an error like this when starting the MCP server:
```
Error: spawn uv ENOENT
```

This indicates that the `uv` package manager cannot be found. Fix this with these steps:

1. Install `uv` using the official installation script:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. If the error persists after installation, create a symlink to make `uv` available system-wide:
   ```bash
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

3. Restart your application or IDE after making these changes.

This ensures that the `uv` executable is accessible from standard system paths that are typically included in the PATH for all processes.
