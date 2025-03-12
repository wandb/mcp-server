"""Utility functions for processing Weave traces."""

import tiktoken
from typing import Any, Dict, List
from datetime import datetime
import re

def truncate_value(value: Any, max_length: int = 200) -> Any:
    """Recursively truncate string values in nested structures."""
    if isinstance(value, str):
        return value[:max_length] + "..." if len(value) > max_length else value
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_length) for k, v in value.items()}
    elif isinstance(value, list):
        return [truncate_value(v, max_length) for v in value]
    return value

def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count if tiktoken fails
        return len(text.split())

def calculate_token_counts(traces: List[Dict]) -> Dict[str, int]:
    """Calculate token counts for traces."""
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    
    for trace in traces:
        input_tokens += count_tokens(str(trace.get("inputs", "")))
        output_tokens += count_tokens(str(trace.get("output", "")))
    
    total_tokens = input_tokens + output_tokens
    
    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "average_tokens_per_trace": round(total_tokens / len(traces), 2) if traces else 0
    }

def generate_status_summary(traces: List[Dict]) -> Dict[str, int]:
    """Generate summary of trace statuses."""
    summary = {"success": 0, "error": 0, "other": 0}
    
    for trace in traces:
        status = trace.get("status", "other").lower()
        if status == "success":
            summary["success"] += 1
        elif status == "error":
            summary["error"] += 1
        else:
            summary["other"] += 1
            
    return summary

def get_time_range(traces: List[Dict]) -> Dict[str, str]:
    """Get the time range of traces."""
    if not traces:
        return {"earliest": None, "latest": None}
    
    dates = []
    for trace in traces:
        started = trace.get("started_at")
        ended = trace.get("ended_at")
        if started:
            dates.append(started)
        if ended:
            dates.append(ended)
    
    if not dates:
        return {"earliest": None, "latest": None}
        
    return {
        "earliest": min(dates),
        "latest": max(dates)
    }

def extract_op_name_distribution(traces: List[Dict]) -> Dict[str, int]:
    """Extract and count the distribution of operation types from Weave URIs.
    
    Converts URIs like 'weave:///wandb-applied-ai-team/mcp-tests/op/query_traces:25DCjPUdNVEKxYOXpQyOCg61XG8GpVZ8RsOlZ6DyouU'
    into a count of operation types like {'query_traces': 5, 'openai.chat.completions.create': 10}
    """
    op_counts = {}
    
    for trace in traces:
        op_name = trace.get("op_name", "")
        if not op_name:
            continue
            
        # Extract the operation name from the URI
        # Pattern matches everything between /op/ and the colon
        match = re.search(r'/op/([^:]+)', op_name)
        if match:
            base_op = match.group(1)
            op_counts[base_op] = op_counts.get(base_op, 0) + 1
    
    # Sort by count in descending order
    return dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True))

def process_traces(
    traces: List[Dict],
    truncate_length: int = 200,
    return_full_data: bool = False
) -> Dict[str, Any]:
    """Process traces and generate metadata."""
    metadata = {
        "total_traces": len(traces),
        "token_counts": calculate_token_counts(traces),
        "time_range": get_time_range(traces),
        "status_summary": generate_status_summary(traces),
        "op_distribution": extract_op_name_distribution(traces)
    }
    
    if return_full_data:
        return {
            "metadata": metadata,
            "traces": traces
        }
    
    truncated_traces = [
        {k: truncate_value(v, truncate_length) for k, v in trace.items()}
        for trace in traces
    ]
    
    return {
        "metadata": metadata,
        "traces": truncated_traces
    } 