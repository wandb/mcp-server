import pytest
import weave
import os
import logging
import json
import glob
import shutil
import time
import uuid # For unique file naming if needed by tests
from datetime import datetime
from wandb_mcp_server.utils import get_rich_logger

os.environ["WANDB_SILENT"] = "True"
weave_logger = get_rich_logger("weave")
weave_logger.setLevel(logging.ERROR)

WANDB_TEST_SUITE_PROJECT = "wandb-mcp-server-test-suite-outputs"
WANDB_TEST_SUITE_ENTITY = "wandb-applied-ai-team"

# Configure logging early
logger = get_rich_logger(__name__)

# Ensure WANDB_SILENT is set if needed
os.environ["WANDB_SILENT"] = "true"

@pytest.fixture(scope="session", autouse=True)
def setup_weave(request):
    """Initializes Weave once per test session - MODIFIED: init call removed."""
    project_name = f"{WANDB_TEST_SUITE_ENTITY}/{WANDB_TEST_SUITE_PROJECT}"
    logger.info(f"Session setup: Weave project target for later init: {project_name}")
    # try:
    #     # weave.init(project_name) # MODIFIED: Call removed from here
    #     # logger.info(f"Weave initialized successfully for {project_name} via setup_weave")
    # except Exception as e:
    #     logger.error(f"Fatal error during Weave initialization in setup_weave: {e}", exc_info=True)
    #     pytest.fail(f"Weave initialization failed in setup_weave: {e}", pytrace=False)

    # Optional: Teardown logic if needed
    # def fin():
    #     logger.info("Tearing down Weave session (if necessary)...")
    # request.addfinalizer(fin)

def pytest_configure(config):
    config.option.asyncio_default_fixture_loop_scope = "function" 

# Assuming weave and EvaluationLogger are importable
# and DateTimeEncoder is available (e.g., from the test_query_weave_traces file or a shared util)
# For simplicity, if DateTimeEncoder is in test_query_weave_traces, we might need to adjust imports
# or duplicate/move it to a shared location. For now, we'll assume it can be handled.
# If DateTimeEncoder is specific to test_query_weave_traces.py, that test file will use it.
# The sessionfinish hook will deserialize standard JSON.

try:
    from weave import EvaluationLogger
except ImportError:
    EvaluationLogger = None

# This should ideally be a shared utility or defined where it can be imported by conftest
# For now, defining a basic one here if not easily importable.
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

WEAVE_RESULTS_DIR_NAME = "weave_eval_results_json"

@pytest.fixture(scope="session")
def weave_results_dir(tmp_path_factory):
    results_dir = tmp_path_factory.mktemp(WEAVE_RESULTS_DIR_NAME, numbered=False)
    yield results_dir

def pytest_sessionfinish(session):
    if not weave or not EvaluationLogger:
        logger.info("Weave SDK not found or EvaluationLogger not imported. Skipping Weave summary logging.")
        return

    logger.info("\nProcessing Weave evaluation results...")
    entity = WANDB_TEST_SUITE_ENTITY
    project = WANDB_TEST_SUITE_PROJECT

    if not entity or not project:
        logger.info("WANDB_TEST_SUITE_ENTITY or WANDB_TEST_SUITE_PROJECT not set. Skipping Weave summary logging.")
        return
    
    try:
        logger.info(f"Initializing Weave EXPLICITLY in session_finish: entity='{entity}', project='{project}'")
        weave.init(f"{entity}/{project}") # This is now the primary init for this process
        logger.info(f"Weave initialized successfully in session_finish for {entity}/{project}")
    except Exception as e:
        logger.error(f"Error initializing Weave in pytest_sessionfinish: {e}. Skipping Weave summary logging.", exc_info=True)
        return

    all_test_data = []
    json_files_found = [] 

    try:
        base_tmp_dir = session.config._tmp_path_factory.getbasetemp()
        logger.info(f"Session base temporary directory: {base_tmp_dir}")
        for item_in_base_tmp_dir in base_tmp_dir.iterdir():
            if item_in_base_tmp_dir.is_dir():
                potential_results_parent_dir = item_in_base_tmp_dir 
                target_results_dir = potential_results_parent_dir / WEAVE_RESULTS_DIR_NAME
                if target_results_dir.is_dir() and target_results_dir.exists():
                    logger.info(f"Searching for JSON files in: {target_results_dir}")
                    json_files_found.extend(list(target_results_dir.glob("*.json")))
                else:
                    if item_in_base_tmp_dir.name == WEAVE_RESULTS_DIR_NAME and item_in_base_tmp_dir.is_dir():
                         logger.info(f"Searching for JSON files in: {item_in_base_tmp_dir}")
                         json_files_found.extend(list(item_in_base_tmp_dir.glob("*.json")))
        json_files_found = sorted(list(set(json_files_found)))
    except Exception as e:
        logger.error(f"Error accessing or searching temporary directories: {e}. Skipping Weave summary logging.", exc_info=True)
        return

    if not json_files_found: 
        logger.info(f"No JSON result files found. Searched in subdirectories of {base_tmp_dir} for '{WEAVE_RESULTS_DIR_NAME}'.")
        return
    
    logger.info(f"Found {len(json_files_found)} JSON result files in total.")

    for json_file_path in json_files_found: 
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                all_test_data.append(data)
        except Exception as e:
            logger.error(f"Error reading or parsing result file {json_file_path}: {e}", exc_info=True)
            continue

    if not all_test_data:
        logger.info("No valid Weave evaluation data could be parsed from JSON files.")
        return

    logger.info(f"DEBUG: Number of items in all_test_data before logging loop: {len(all_test_data)}")
    if all_test_data:
        logger.info(f"DEBUG: Metadata of first item for logging: {all_test_data[0].get('metadata')}")
    if len(all_test_data) > 1:
        logger.info(f"DEBUG: Metadata of second item for logging: {all_test_data[1].get('metadata')}")

    run_timestamp = int(time.time()) 
    git_commit_id_from_tests = "unknown_git_commit_id"
    source_test_file_name_cleaned = "unknown_test_file"

    first_valid_metadata = all_test_data[0].get('metadata', {})
    if first_valid_metadata.get('git_commit_id'):
        git_commit_id_from_tests = first_valid_metadata['git_commit_id']
    if first_valid_metadata.get('source_test_file_name'):
        raw_file_name = first_valid_metadata['source_test_file_name']
        source_test_file_name_cleaned = raw_file_name.replace('.py', '').replace('_', '-')
    
    overall_eval_name = f"mcp-eval_{source_test_file_name_cleaned}_{git_commit_id_from_tests}_{run_timestamp}"
    aggregated_dataset_name = f"{source_test_file_name_cleaned}_tests"

    logger.info(f"Logging to Weave Evaluation: Name='{overall_eval_name}', Git Commit='{git_commit_id_from_tests}', Dataset='{aggregated_dataset_name}'")

    try:
        session_eval_logger = EvaluationLogger(
            name=overall_eval_name,
            model=git_commit_id_from_tests, 
            dataset=aggregated_dataset_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize session EvaluationLogger: {e}. Collected data will not be logged to Weave.", exc_info=True)
        return

    total_tests_logged = 0
    passed_tests_logged = 0
    all_latencies = [] 

    for test_data in all_test_data:
        try:
            metadata = test_data.get('metadata', {})
            inputs = test_data.get('inputs', {})
            output = test_data.get('output', {})
            score_value = test_data.get('score', False)
            scorer_name = test_data.get('scorer_name', 'test_outcome')
            metrics = test_data.get('metrics', {}) 
            execution_latency = metrics.get("execution_latency_seconds") 

            if 'test_case_index' in metadata:
                inputs['_test_case_index'] = metadata['test_case_index']
            if 'sample_name' in metadata:
                inputs['_sample_name'] = metadata['sample_name']
            if 'source_test_file_name' in metadata:
                inputs['_source_test_file_name'] = metadata['source_test_file_name']

            pred_logger = session_eval_logger.log_prediction(
                inputs=inputs,
                output=output
            )
            pred_logger.log_score(scorer=scorer_name, score=bool(score_value))
            if execution_latency is not None:
                pred_logger.log_score(scorer="execution_latency_seconds", score=float(execution_latency))
                all_latencies.append(float(execution_latency))
            
            pred_logger.finish()

            total_tests_logged += 1
            if score_value:
                passed_tests_logged += 1
        except Exception as e:
            example_id_for_error_msg = str(metadata.get('test_case_index', metadata.get('sample_name', 'unknown_example')))
            logger.error(f"Error logging prediction to Weave for example identified by '{example_id_for_error_msg}': {e}", exc_info=True)

    if total_tests_logged > 0:
        summary_metrics = {
            "total_tests_processed_from_files": len(all_test_data),
            "total_tests_logged_to_weave": total_tests_logged,
            "passed_tests_logged_to_weave": passed_tests_logged,
            "pass_rate": (passed_tests_logged / total_tests_logged),
            "session_worker_count": str(session.config.option.numprocesses if hasattr(session.config.option, 'numprocesses') else 'N/A')
        }
        if all_latencies:
            summary_metrics["avg_execution_latency_seconds"] = sum(all_latencies) / len(all_latencies)
            summary_metrics["min_execution_latency_seconds"] = min(all_latencies)
            summary_metrics["max_execution_latency_seconds"] = max(all_latencies)
            summary_metrics["total_execution_latency_seconds"] = sum(all_latencies)

        logger.info(f"Final Weave summary to log: {summary_metrics}")
        try:
            session_eval_logger.log_summary(summary_metrics)
            logger.info(f"Successfully logged Weave summary for '{overall_eval_name}' to W&B project '{entity}/{project}'")
        except Exception as e:
            logger.error(f"Failed to log Weave summary for '{overall_eval_name}': {e}", exc_info=True)
    else:
        logger.info("No tests were successfully logged to Weave, skipping summary.")