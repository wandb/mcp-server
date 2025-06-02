<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>

# Weights & Biases MCP Server

A Model Context Protocol (MCP) server for querying [Weights & Biases](https://www.wandb.ai/) data. This server allows a MCP Client to:

- query W&B Models runs and sweeps
- query W&B Weave traces, evaluations and datasets
- query [wandbot](https://github.com/wandb/wandbot), the W&B support agent, for general W&B feature questions
- run python code in isolated E2B or Pyodide sandboxes for data analysis
- write text and charts to W&B Reports


## Installation

### Install `uv`

Please first [install `uv`](https://docs.astral.sh/uv/getting-started/installation/) with either:


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or 

```bash
brew install uv
```

### Code sandbox setup

The wandb MCP server exposes a secure, isolated python code sandbox tool to the client to let it send code (e.g. pandas) for additional data analysis to be run on queried W&B data. 

**Option 1: Local Pyodide sandbox - Install Deno**

The local Pyodide sandbox uses Deno to isolate execution from the host system. This will be used if an E2B api is not found, just install Deno to enable this option:

```bash
# One-line install for macOS/Linux:
curl -fsSL https://deno.land/install.sh | sh

# Or on Windows (PowerShell):
irm https://deno.land/install.ps1 | iex
```

Deno will automatically download the Pyodide runtime when first used.

**Option 2: Hosted E2B sandbox - Set E2B api key**

The sandbox tool will default to E2B is an E2B api key is detected. To use the hosted [E2B](https;//www.e2b.dev) sandbox:

1. Sign up to E2B at [e2b.dev](https://e2b.dev)
2. Set the `E2B_API_KEY` environment variable

If neither Deno nor an E2B api key are detected then the `execute_sandbox_code_tool` tool will not be added to the wandb MCP server. To explicitly disable the sandbox tool then explicitly set the `DISABLE_CODE_SANDBOX=1` environment variable.


### Installation helpers

We provide a helper utility below to easily install the Weights & Biases MCP Server into applications that use a JSON server spec - inspired by the OpenMCP Server Registry [add-to-client pattern](https://www.open-mcp.org/servers).


### Cursor installation
#### Specific Cursor project
Enabel the MCP for a specific project. Run the following in the root of your project dir:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client .cursor/mcp.json && uvx wandb login
```

#### Cursor global
Enable the MCP for all Curosor projects, doesn't matter where this is run:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/.cursor/mcp.json && uvx wandb login
```

### Windsurf installation

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/.codeium/windsurf/mcp_config.json && uvx wandb login
```

### Claude Desktop installation
First ensure `uv` is installed, you might have to use brew to install depite `uv` being available in your terminal.

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/Library/Application\ Support/Claude/claude_desktop_config.json && uvx wandb login
```

## Manual Installation
1. Ensure you have `uv` installed, see above installation instructions for uv.
2. Get your W&B api key [here](https://www.wandb.ai/authorize)
3. Add the following to your MCP client config manually.

```bash
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server"
      ]
      "envs": {
        "WANDB_API_KEY": <insert your wandb key>
      }
    }
  }
}
```

### Runing from Source

Run the server from source by running the below in the root dir:

```bash
wandb login && uv run src/wandb_mcp_server/server.py
```


## Available MCP tools

### wandb
-  **`query_wandb_tool`** Execute queries against wandb experiment tracking data including Runs & Sweeps.
  
### weave
- **`query_weave_traces_tool`** Queries Weave traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **`count_weave_traces_tool`** Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.

### W&B Support agent
- **`query_wandb_support_bot`** Connect your client to [wandbot](https://github.com/wandb/wandbot), our RAG-powered support agent for general help on how to use Weigths & Biases products and features.

### Python code sandbox
- **`execute_sandbox_code_tool`** Execute Python code in secure, isolated sandbox environments, either a hosted E2B sandbox or a local Pyodide sandbox, WebAssembly-based execution that uses Deno to isolate execution from the host system (inspired by [Pydantic AI's Run Python MCP](https://ai.pydantic.dev/mcp/run-python/)). See sandbox setup instructions above.

### Saving Analysis
- **`create_wandb_report_tool`** Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

### General W&B helpers
- **`query_wandb_entity_projects`** List the available W&B entities and projects that can be accessed to give the LLM more context on how to write the correct queries for the above tools.

## Sandbox Configuration (Optional)

You can configure sandbox behavior using environment variables:

#### Disable Sandbox
- `DISABLE_CODE_SANDBOX`: Set to any value to completely disable the code sandbox tool (e.g., `DISABLE_CODE_SANDBOX=1`)

#### Package Installation Security
Control which packages can be installed in E2B sandboxes:
- `E2B_PACKAGE_ALLOWLIST`: Comma-separated list of allowed packages (e.g., `numpy,pandas,matplotlib`)
- `E2B_PACKAGE_DENYLIST`: Comma-separated list of denied packages (default includes potentially dangerous packages)

#### Cache Settings
- `E2B_CACHE_TTL_SECONDS`: Execution cache TTL in seconds (default: 900 = 15 minutes)

## Usage tips

#### Provide your W&B project and entity name

LLMs are not mind readers, ensure you specify the W&B Entity and W&B Project to the LLM. Example query for Claude Desktop:

```markdown
how many openai.chat traces in the wandb-applied-ai-team/mcp-tests weave project? plot the most recent 5 traces over time and save to a report
```

#### Avoid asking overly broad questions

Questions such as "what is my best evaluation?" are probably overly broad and you'll get to an answer faster by refining your question to be more specific such as: "what eval had the highest f1 score?"

#### Ensure all data was retrieved

When asking broad, general questions such as "what are my best performing runs/evaluations?" its always a good idea to ask the LLM to check that it retrieved all the available runs. The MCP tools are designed to fetch the correct amount of data, but sometimes there can be a tendency from the LLMs to only retrieve the latest runs or the last N runs.



## Troubleshooting

### Authentication

Ensure the machine running the MCP server is authenticated  to Weights & Biases, either by setting the `WANDB_API_KEY` or running the below to add the key to the .netrc file:

```bash
uvx wandb login
```

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


## Testing

The tests include a mix of unit tests and integration tests that test the tool calling reliability of a LLM. For now the integration tets only use claude-sonnet-3.7.


####Set LLM provider API key

Set the appropriate api key in the `.env` file, e.g.

```
ANTHROPIC_API_KEY=<my_key>
```

####Run 1 test file

Run a single test using pytest with 10 workers
```
uv run pytest -s -n 10 tests/test_query_wandb_gql.py
```

####Test debugging

Turn on debug logging for a single sample in 1 test file

```
pytest -s -n 1 "tests/test_query_weave_traces.py::test_query_weave_trace[longest_eval_most_expensive_child]" -v --log-cli-level=DEBUG
```
