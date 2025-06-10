import json
import logging
import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import simple_parsing

from wandb_mcp_server.utils import get_rich_logger

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = get_rich_logger(__name__)


def find_deno_installation() -> Optional[str]:
    """
    Find Deno installation and return the path to the deno binary.
    
    Returns:
        Path to deno binary if found, None otherwise
    """
    import platform
    
    system = platform.system().lower()
    
    # Common Deno installation paths across platforms
    potential_paths = []
    
    if system == "windows":
        # Windows-specific paths
        home = os.path.expanduser("~")
        potential_paths.extend([
            # Default PowerShell install location
            os.path.join(home, ".deno", "bin", "deno.exe"),
            # Scoop install location
            os.path.join(home, "scoop", "apps", "deno", "current", "deno.exe"),
            # Chocolatey install location
            "C:\\ProgramData\\chocolatey\\bin\\deno.exe",
            # Winget/system install locations
            "C:\\Program Files\\deno\\deno.exe",
            "C:\\Program Files (x86)\\deno\\deno.exe",
            # npm global install (Windows)
            os.path.join(home, "AppData", "Roaming", "npm", "deno.exe"),
            # System PATH locations
            "C:\\Windows\\System32\\deno.exe",
            "C:\\Windows\\deno.exe",
        ])
    else:
        # Unix-like systems (macOS, Linux)
        home = os.path.expanduser("~")
        potential_paths.extend([
            # Default shell script install location
            os.path.join(home, ".deno", "bin", "deno"),
            # Homebrew (macOS)
            "/opt/homebrew/bin/deno",  # Apple Silicon
            "/usr/local/bin/deno",     # Intel Mac / Linux Homebrew
            # MacPorts (macOS)
            "/opt/local/bin/deno",
            # System package manager locations (Linux)
            "/usr/bin/deno",
            "/bin/deno",
            "/usr/sbin/deno",
            "/sbin/deno",
            # Snap (Linux)
            "/snap/bin/deno",
            # Flatpak (Linux)
            "/var/lib/flatpak/exports/bin/deno",
            os.path.join(home, ".local", "share", "flatpak", "exports", "bin", "deno"),
            # asdf version manager
            os.path.join(home, ".asdf", "shims", "deno"),
            # vfox version manager  
            os.path.join(home, ".version-fox", "cache", "deno", "current", "bin", "deno"),
            # Nix
            os.path.join(home, ".nix-profile", "bin", "deno"),
            "/nix/var/nix/profiles/default/bin/deno",
            # Cargo install location
            os.path.join(home, ".cargo", "bin", "deno"),
            # npm global install (Unix)
            os.path.join(home, ".npm-global", "bin", "deno"),
            "/usr/local/lib/node_modules/.bin/deno",
            # Manual install locations
            os.path.join(home, "bin", "deno"),
            os.path.join(home, ".local", "bin", "deno"),
            # AppImage (Linux)
            os.path.join(home, "Applications", "deno"),
            "/opt/deno/bin/deno",
        ])
    
    for deno_path in potential_paths:
        if os.path.isfile(deno_path) and os.access(deno_path, os.X_OK):
            try:
                # Verify it's actually a working Deno installation
                result = subprocess.run(
                    [deno_path, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"✅ Found working Deno installation at: {deno_path}")
                    return os.path.dirname(deno_path)  # Return the directory
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"Deno at {deno_path} failed verification: {e}")
                continue
    
    return None


def check_sandbox_requirements() -> Dict[str, bool]:
    """
    Check if sandbox requirements are met and provide helpful guidance.
    
    Returns:
        Dict with availability status for each sandbox type
    """
    status = {"deno_available": False, "e2b_available": False, "deno_path": None}
    
    # Check Deno availability
    try:
        result = subprocess.run(
            ["deno", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            status["deno_available"] = True
            logger.info("✅ Deno detected in PATH - Pyodide sandbox will be available")
        else:
            logger.warning("⚠️  Deno command failed - Pyodide sandbox may not work")
    except FileNotFoundError:
        # Deno not in PATH, try to find it
        deno_dir = find_deno_installation()
        if deno_dir:
            status["deno_path"] = deno_dir
            logger.warning("⚠️  Deno not in PATH but found at alternative location")
            logger.info(f"   Found Deno at: {deno_dir}")
            logger.info("   Consider using --add-deno-path to automatically add to MCP config")
        else:
            logger.warning("⚠️  Deno not found - Pyodide sandbox will not be available")
            logger.info("   To install Deno: curl -fsSL https://deno.land/install.sh | sh -s -- -y")
            logger.info("   Then restart your terminal or run: source ~/.bashrc")
            logger.info("   If issues persist, see troubleshooting in the README")
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Deno check timed out - may indicate installation issues")
    except Exception as e:
        logger.warning(f"⚠️  Error checking Deno: {e}")
    
    # Check E2B availability
    if os.getenv("E2B_API_KEY"):
        status["e2b_available"] = True
        logger.info("✅ E2B API key detected - E2B sandbox will be available")
    else:
        logger.info("ℹ️  E2B API key not set - E2B sandbox will not be available")
        logger.info("   To enable E2B: Get API key from https://e2b.dev and set E2B_API_KEY")
    
    # Summary guidance
    if not status["deno_available"] and not status["e2b_available"]:
        logger.warning("⚠️  No sandbox environments detected!")
        logger.info("   Code execution tools will not be available.")
        if status["deno_path"]:
            logger.info("   💡 Tip: Use --add-deno-path to automatically fix Deno PATH issues")
        else:
            logger.info("   Consider setting up Deno (local) or E2B (cloud) for code execution.")
    elif status["deno_available"] or status["e2b_available"]:
        available_types = []
        if status["deno_available"]:
            available_types.append("Pyodide (local)")
        if status["e2b_available"]:
            available_types.append("E2B (cloud)")
        logger.info(f"✅ Code execution available via: {', '.join(available_types)}")
    
    return status


@dataclass
class AddToClientArgs:
    """Add Weights & Biases MCP server to your client config."""

    config_path: str
    """Path to the MCP client config file"""

    wandb_api_key: Optional[str] = None
    """W&B API key for authentication"""

    e2b_api_key: Optional[str] = None
    """E2B API key for cloud sandbox execution"""

    disable_code_sandbox: Optional[str] = None
    """Set to any value to disable code sandbox (e.g., '1' or 'true')"""

    add_deno_path: bool = False
    """Automatically add Deno path to PATH environment variable if Deno is found"""

    write_env_vars: List[str] = field(default_factory=list)
    """Write additional environment variables to client config file (format: KEY=VALUE)"""

    def get_env_vars(self) -> Dict[str, str]:
        """Get all environment variables to include in the config."""
        env_vars = {}

        # Parse additional env vars from list
        for env_str in self.write_env_vars:
            if "=" in env_str:
                key, value = env_str.split("=", 1)
                env_vars[key] = value

        # Add specific environment variables if provided
        if self.wandb_api_key:
            env_vars["WANDB_API_KEY"] = self.wandb_api_key

        if self.e2b_api_key:
            env_vars["E2B_API_KEY"] = self.e2b_api_key

        if self.disable_code_sandbox:
            env_vars["DISABLE_CODE_SANDBOX"] = self.disable_code_sandbox

        # Add Deno PATH if requested and Deno is found
        if self.add_deno_path:
            deno_dir = find_deno_installation()
            if deno_dir:
                current_path = os.environ.get("PATH", "")
                # Common system paths to include
                system_paths = [
                    "/usr/local/bin",
                    "/opt/homebrew/bin", 
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin"
                ]
                
                # Combine Deno path with current PATH and system paths
                all_paths = [deno_dir] + current_path.split(":") + system_paths
                
                # Remove duplicates while preserving order
                unique_paths = []
                for path in all_paths:
                    if path and path not in unique_paths:
                        unique_paths.append(path)
                
                env_vars["PATH"] = ":".join(unique_paths)
                logger.info(f"✅ Added Deno path to MCP config: {deno_dir}")
            else:
                logger.warning("⚠️  --add-deno-path requested but Deno not found")
                logger.info("   Install Deno first: curl -fsSL https://deno.land/install.sh | sh -s -- -y")

        return env_vars


def get_new_config(env_vars: Optional[Dict[str, str]] = None) -> dict:
    """
    Get the new configuration to add to the client config.

    Args:
        env_vars: Optional environment variables to include in the config

    Returns:
        Dictionary with the MCP server configuration
    """
    config = {
        "mcpServers": {
            "wandb": {
                "command": "uvx",
                "args": [
                    "--from",
                    "git+https://github.com/wandb/wandb-mcp-server",
                    "wandb_mcp_server",
                ],
            }
        }
    }

    # Add environment variables if provided
    if env_vars:
        config["mcpServers"]["wandb"]["env"] = env_vars

    return config


def add_to_client(args: AddToClientArgs) -> None:
    """
    Add MCP server configuration to a client config file.

    Args:
        args: Command line arguments

    Raises:
        Exception: If there are errors reading/writing the config file
    """
    # Handle potential path parsing issues
    config_path = args.config_path
    
    # Debug: Log the raw config_path to help diagnose issues
    logger.debug(f"Raw config_path argument: '{config_path}'")
    
    # Check if config_path looks malformed (starts with --)
    if config_path.startswith("--"):
        logger.error(f"Invalid config path detected: '{config_path}'")
        logger.error("This usually happens when command line arguments are not properly parsed.")
        logger.error("Try running the command on a single line or check for syntax errors.")
        sys.exit(1)
    
    # Expand user path and resolve to absolute path
    config_path = os.path.expanduser(config_path)
    config_path = os.path.abspath(config_path)
    
    logger.info(f"Using config path: {config_path}")

    # Read existing config file or initialize a default structure
    config = {"mcpServers": {}}  # Start with a default, ensures mcpServers key exists
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                # Attempt to load. If file is empty or has invalid JSON,
                # json.load will raise JSONDecodeError.
                loaded_config = json.load(f)
                # If load is successful, check if it's a dictionary (top-level JSON should be an object)
                if isinstance(loaded_config, dict):
                    config = loaded_config  # Use the loaded config
                    logger.info(f"Loaded existing config from {config_path}")
                else:
                    # Loaded JSON is not a dictionary (e.g. `null`, `[]`, `true`)
                    # This is unexpected for a config file that should hold mcpServers.
                    logger.warning(
                        f"Config file {config_path} did not contain a JSON object. Using default config."
                    )
                    # config remains the default {"mcpServers": {}}
        else:
            logger.info(
                f"Config file {config_path} doesn't exist. Will create new file."
            )
            # config remains the default {"mcpServers": {}}
    except json.JSONDecodeError as e:
        # This handles empty file or malformed JSON.
        logger.warning(
            f"Config file {config_path} is empty or contains invalid JSON: {e}. Using default config."
        )
        # config remains the default {"mcpServers": {}}.
    except IOError as e:
        logger.error(
            f"Fatal error reading config file {config_path}: {e}. Cannot proceed."
        )
        sys.exit(f"Fatal error reading config file: {e}")  # Exit if we can't read

    if not isinstance(config.get("mcpServers"), dict):
        if os.path.exists(config_path):
            logger.warning(
                f"Warning: 'mcpServers' key in the loaded config from {config_path} was missing or not a dictionary. Initializing it."
            )
        config["mcpServers"] = {}  # Ensure it's a dictionary

    # Get the new configuration with environment variables
    env_vars = args.get_env_vars()
    new_config = get_new_config(env_vars)

    # Check for key overlaps
    existing_keys = set(config["mcpServers"].keys())
    new_keys = set(new_config["mcpServers"].keys())
    overlapping_keys = existing_keys.intersection(new_keys)

    if overlapping_keys:
        logger.info(
            "The following tools already exist in your config and will be overwritten:"
        )
        for key in overlapping_keys:
            logger.info(f"- {key}")

        # Ask for confirmation
        answer = input("Do you want to overwrite them? (y/N): ").lower()
        if answer != "y":
            logger.info("Operation cancelled.")
            sys.exit(0)

    # Update config with new servers
    config["mcpServers"].update(new_config["mcpServers"])

    # Create directory if it doesn't exist
    config_dir = os.path.dirname(config_path)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Save the updated config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Successfully updated config at {config_path}")
    
    # Check sandbox requirements and provide guidance
    logger.info("\n" + "="*50)
    logger.info("Checking sandbox requirements...")
    check_sandbox_requirements()
    logger.info("="*50)


def add_to_client_cli():
    args = simple_parsing.parse(AddToClientArgs)
    add_to_client(args)


if __name__ == "__main__":
    add_to_client_cli()
