# Weave MCP Server

A Model Context Protocol (MCP) server for querying Weave traces. This server allows you to interact with Weave traces through various MCP clients.

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
uv run src/weave_mcp_server/server.py
```

## Client Setup

### Cursor

Add the server configuration to your Cursor settings:

1. Open Cursor settings
2. Navigate to the MCP section
3. Add a new server configuration:
```json
{
    "weave-trace-mcp-server": {
        "command": "uv",
        "args": [
            "--directory",
            "/ABSOLUTE/PATH/TO/YOUR/PROJECT",
            "run",
            "src/weave_mcp_server/server.py"
        ]
    }
}
```

### Cline

Configure the server in Cline's settings:

1. Open Cline's configuration file
2. Add the server to the MCP servers section:
```json
{
    "mcpServers": {
        "weave-trace-mcp-server": {
            "command": "uv run src/weave_mcp_server/server.py",
            "cwd": "/ABSOLUTE/PATH/TO/YOUR/PROJECT"
        }
    }
}
```

### Claude for Desktop

Configure Claude for Desktop by editing `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "weave-trace-mcp-server": {
            "command": "uv",
            "args": [
                "--directory",
                "/ABSOLUTE/PATH/TO/YOUR/PROJECT",
                "run",
                "src/weave_mcp_server/server.py"
            ]
        }
    }
}
```

## Available Tools

The server provides the following tool:

- `query_traces_tool`: Query Weave traces with filtering and sorting options
  - Parameters:
    - `entity_name`: The Weights & Biases entity name (team or username)
    - `project_name`: The Weights & Biases project name
    - `filters`: Optional filters for traces (display_name, op_name, trace_id, etc.)
    - `sort_by`: Field to sort by (started_at, ended_at, op_name, etc.)
    - `sort_direction`: Sort direction (asc or desc)
    - `limit`: Maximum number of results to return
    - `offset`: Number of results to skip (for pagination)

## Troubleshooting

If the server isn't showing up in your MCP client:

1. Check the client's logs for MCP-related errors
2. Verify your configuration file syntax
3. Make sure the path to your project is absolute
4. Restart the client application

For detailed logs, check the client's log directory. Common locations:
- Claude for Desktop: `~/Library/Logs/Claude/mcp*.log`
- Cursor: Check the Cursor logs panel
- Cline: Check the Cline output window

## API Endpoints

### Health Check

```