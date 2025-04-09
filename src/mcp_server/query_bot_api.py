import os

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


def query_support_bot_api(question):
    BASE_URL = os.getenv("WANDBOT_BASE_URL")
    QUERY_ENDPOINT = f"{BASE_URL}/chat/query"
    STATUS_ENDPOINT = f"{BASE_URL}/status"
    QUERY_TIMEOUT_SECONDS = 40
    STATUS_TIMEOUT_SECONDS = 10

    try:
        status_response = requests.get(
            STATUS_ENDPOINT,
            headers={"Accept": "application/json"},
            timeout=STATUS_TIMEOUT_SECONDS,
        )

        # Check HTTP status code
        status_response.raise_for_status()

        # Try to parse JSON, handle potential parsing errors
        try:
            status_result = status_response.json()
        except ValueError:
            return {
                "answer": "Error: Unable to parse response from support bot.",
                "sources": [],
            }

        # Validate expected response structure
        if "initialized" not in status_result:
            return {
                "answer": "Error: Received unexpected response format from support bot.",
                "sources": [],
            }

        if status_result["initialized"]:
            try:
                response = requests.post(
                    QUERY_ENDPOINT,
                    headers={"Content-Type": "application/json"},
                    json={
                        "question": question,
                        "application": "wandb_mcp_server",
                    },
                    timeout=QUERY_TIMEOUT_SECONDS,
                )

                # Check HTTP status code
                response.raise_for_status()

                # Try to parse JSON, handle potential parsing errors
                try:
                    result = response.json()
                except ValueError:
                    return {
                        "answer": "Error: Unable to parse response data from support bot.",
                        "sources": [],
                    }

                # Validate expected response structure
                if "answer" not in result or "sources" not in result:
                    return {
                        "answer": "Error: Received incomplete response from support bot.",
                        "sources": [],
                    }

                return {"answer": result["answer"], "sources": result["sources"]}

            except requests.Timeout:
                return {
                    "answer": "Error: Support bot request timed out. Please try again later.",
                    "sources": [],
                }
            except requests.RequestException as e:
                return {
                    "answer": f"Error connecting to support bot: {str(e)}",
                    "sources": [],
                }
        else:
            return {
                "answer": "The support bot is appears to be offline. Please try again later.",
                "sources": [],
            }

    except requests.Timeout:
        return {
            "answer": "Error: Support bot status check timed out. Please try again later.",
            "sources": [],
        }
    except requests.RequestException as e:
        return {
            "answer": f"Error connecting to support bot: {str(e)}",
            "sources": [],
        }
