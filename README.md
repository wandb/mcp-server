# MCP
A collection of MCP (Model Context Protocol) tools and examples for wandb and weave


## Basic Server Setup
```
# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch <server_file_name>.py
```

Create your MCP server file


Use `uv` to run the server:

```
uv run <server_file_name>.py
```