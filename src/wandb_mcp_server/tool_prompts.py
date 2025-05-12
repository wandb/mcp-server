LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """
Fetch all projects for a specific wandb or weave entity. Useful to use when 
the user hasn't specified a project name or queries are failing due to a 
missing or incorrect Weights & Biases project name.

If no entity is provided, the tool will fetch all projects for the current user 
as well as all the project in the teams they are part of.

<critical_info>

**Important:**

Do not use this tool if the user has not specified a W&BB entity name. Instead ask
the user to provide either their W&B username or W&B team name.
</critical_info>

<debugging_tips>

**Error Handling:**

If this function throws an error, it's likely because the W&B entity name is incorrect.
If this is the case, ask the user to double check the W&B entity name given by the user, 
either their personal user or their W&B Team name.

**Expected Project Name Not Found:**

If the user doesn't see the project they're looking for in the list of projects,
ask them to double check the W&B entity name, either their personal W&B username or their 
W&B Team name.
</debugging_tips>

Args:
    entity (str): The wandb entity (username or team name)
    
Returns:
    List[Dict[str, Any]]: List of project dictionaries containing:
        - name: Project name
        - entity: Entity name
        - description: Project description
        - visibility: Project visibility (public/private)
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - tags: List of project tags
"""

QUERY_WEAVE_TRACES_TOOL_DESCRIPTION = """
Query Weave traces, trace metadata, and trace costs with filtering and sorting options.

<wandb_vs_weave_product_distinction>
**IMPORTANT PRODUCT DISTINCTION:**
W&B offers two distinct products with different purposes:

1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
    lifecycle management. Use `query_wandb_gql_tool` for questions about:
    - Experiment runs, metrics, and performance comparisons
    - Artifact management and model registry
    - Hyperparameter optimization and sweeps
    - Project dashboards and reports

2. W&B Weave: A toolkit for LLM and GenAI application observability and evaluation. Use
    `query_weave_traces_tool` (this tool) for questions about:
    - Execution traces and paths of LLM operations
    - LLM inputs, outputs, and intermediate results
    - Chain of thought visualization and debugging
    - LLM evaluation results and feedback
</wandb_vs_weave_product_distinction>

<use_case_selector>
**USE CASE SELECTOR - READ FIRST:**
- For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_gql_tool
- For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

=====================================================================
⚠️ TOOL SELECTION WARNING ⚠️
This tool is ONLY for WEAVE TRACES (LLM operations), NOT for run metrics or experiments!
=====================================================================

**KEYWORD GUIDE:**
If user question contains:
- "runs", "experiments", "metrics" → Use query_wandb_gql_tool
- "traces", "LLM calls" etc → Use this tool

**COMMON MISUSE CASES:**
❌ "Looking at metrics of my latest runs" - Do NOT use this tool, use query_wandb_gql_tool instead
❌ "Compare performance across experiments" - Do NOT use this tool, use query_wandb_gql_tool instead
</use_case_selector>

If the users asks for data about "runs" or "experiments" or anything about "experiment tracking"
then use the `query_wandb_gql_tool` instead.
</use_case_selector>

<usage_tips>
query_traces_tool can return a lot of data, below are some usage tips for this function
in order to avoid overwhelming a LLM's context window with too much data.

<managing_llm_context_window>

Returning all weave trace data can possibly result in overwhelming the LLM context window
if there are 100s or 1000s of logged weave traces (depending on how many child traces each has) as
well as resulting in a lot of data from or calls to the weave API.

So, depending on the user query, consider doing the following to return enough data to answer the user query
but not too much data that it overwhelms the LLM context window:

- return only the root traces using the `trace_roots_only` boolean filter if you only need the top-level/parent
traces and don't need the data from all child traces. For example, if a user wants to know the number of
successful traces in a project but doesn't need the data from all child traces. Or if a user
wants to visualise the number of parent traces over time.

- return only the truncated values of the trace data keys in order to first give a preview of the data that can then
inform more targeted weave trace queries from the user. in the extreme you can set `truncate_length` to 0 in order to
only return keys but not the values of the trace data.

- return only the metadata for all the traces (set `metadata_only = True`) if the query doesn't need to know anything
about the structure or content of the individual weave traces. Note that this still requires
requesting all the raw traces data from the weave API so can still result in a lot of data and/or a
lot of calls being made to the weave API.

- return only the columns needed using the `columns` parameter. In weave, the `inputs` and `output` columns of a
trace can contain a lot of data, so avoiding returning these columns can help. Note you have to explicitly specify
the columns you want to return if there are certain columns you don't want to return. Its almost always a good idea to
specficy the columns needed.

<returning_metadata_only>

If `metadata_only = True` this returns only metadata of the traces such as trace counts, token counts,
trace types, time range, status counts and distribution of op names. if `metadata_only = False` the
trace data is returned either in full or truncated to `truncate_length` characters depending if
`return_full_data = True` or `False` respectively.
</returning_metadata_only>

<truncating_trace_data_values>

If `return_full_data = False` the trace data is truncated to `truncate_length` characters,
default 200 characters. Otherwise the trace data is returned in full.
</truncating_trace_data_values>

Remember, LLM context window is precious, only return the minimum amount of data needed to complete an analysis.
</managing_llm_context_window>

<usage_guidance>

- Exploratory queries: For generic exploratory or initial queries about a set of weave traces in a project it can
be a good idea to start with just returning metadata or truncated data. Consider asking the
user for clarification and warn them that returning a lot of weave traces data might
overwhelm the LLM context window. No need to warn them multiple times, just once is enough.

- Project size: Consider using the count_traces_tool to get an estimate of the number of traces in a project
before querying for them as query_trace_tool can return a lot of data.

- Partial op name matching: Use the `op_name_contains` filter if a users has only given a partial op name or if they
are unsure of the exact op name.

- Evaluations: If asked about weave evaluations or evals traces filter for traces with:
    `op_name_contains = "Evaluation.evaluate"` as a first step. These ops are parent traces that contain
    aggregated stats and scores about the evaluation. The child traces of these ops are the actual evaluation results
    for each sample in an evaluation dataset. If asked about individual rows in an evaluation then use the parent_ids
    filter to return the child traces.

- Weave nomenclature: Note that users might refer to weave ops as "traces" or "calls" or "traces" as "ops".

</usage_guidance>
</usage_tips>

Args:
    entity_name: The Weights & Biases entity name (team or username)
    project_name: The Weights & Biases project name
    filters: Dict of filter conditions, supporting:
        - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
        - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
        - op_name_contains: Filter for op_name containing this substring (easier than regex)
        - trace_roots_only: Boolean to filter for only top-level/parent traces. Useful when you don't need
            to return the data from all child traces.
        - trace_id: Filter by specific trace ID
        - call_ids: Filter by specific call IDs (string or list of strings). Note it is the call_id, not the
            trace id that is exposed in a weave url.
        - parent_ids: Return traces that are children of the given parent trace ids (string or list of strings)
        - status: Filter by trace status, defined as whether or not the trace had an exception or not. Can be
            `success` or `exception`.
        - time_range: Dict with "start" and "end" datetime strings. Datetime strings should be in ISO format
            (e.g. `2024-01-01T00:00:00Z`)
        - attributes: Dict of the weave attributes of the trace.
            Supports nested paths (e.g., "metadata.model_name") via dot notation.
            Value can be:
            *   A literal for exact equality (e.g., `"status": "success"`)
            *   A dictionary with a comparison operator: `$gt`, `$lt`, `$eq`, `$gte`, `$lte` (e.g., `{"token_count": {"$gt": 100}}`)
            *   A dictionary with the `$contains` operator for substring matching on string attributes (e.g., `{"model_name": {"$contains": "gpt-3"}}`)
            **Warning:** The `$contains` operator performs simple substring matching only, full regular expression matching (e.g., via `$regex`) is **not supported** for attributes. Do not attempt to use `$regex`.
        - has_exception: Optional[bool] to filter traces by exception status:
            - None (or key not present): Show all traces regardless of exception status
            - True: Show only traces that have exceptions (exception field is not null)
            - False: Show only traces without exceptions (exception field is null)
    sort_by: Field to sort by (started_at, ended_at, op_name, etc.). Defaults to 'started_at'
    sort_direction: Sort direction ('asc' or 'desc'). Defaults to 'desc'
    limit: Maximum number of results to return. Defaults to None
    include_costs: Include tracked api cost information in the results. Defaults to True
    include_feedback: Include weave annotations (human labels/feedback). Defaults to True
    columns: List of specific columns to include in the results. Its almost always a good idea to specficy the
    columns needed. Defaults to None (all columns).
        Available columns are:
            id: <class 'str'>
            project_id: <class 'str'>
            op_name: <class 'str'>
            display_name: typing.Optional[str]
            trace_id: <class 'str'>
            parent_id: typing.Optional[str]
            started_at: <class 'datetime.datetime'>
            attributes: dict[str, typing.Any]
            inputs: dict[str, typing.Any]
            ended_at: typing.Optional[datetime.datetime]
            exception: typing.Optional[str]
            output: typing.Optional[typing.Any]
            summary: typing.Optional[SummaryMap]
            wb_user_id: typing.Optional[str]
            wb_run_id: typing.Optional[str]
            deleted_at: typing.Optional[datetime.datetime]
    expand_columns: List of columns to expand in the results. Defaults to None
    truncate_length: Maximum length for string values in weave traces. Defaults to 200
    return_full_data: Whether to include full untruncated trace data. Defaults to False
    metadata_only: Return only metadata without traces. Defaults to False

Returns:
    JSON string containing either full trace data or metadata only, depending on parameters

<examples>
    ```python
    # Get an overview of the traces in a project
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"root_traces_only": True},
        metadata_only=True,
        return_full_data=False
    )

    # Get failed traces with costs and feedback
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"status": "error"},
        include_costs=True,
        include_feedback=True
    )

    # Get specific columns for traces who's op name (i.e. trace name) contains a specific substring
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"op_name_contains": "Evaluation.summarize"},
        columns=["id", "op_name", "started_at", "costs"]
    )
    ```
</examples>
"""


CREATE_WANDB_REPORT_TOOL_DESCRIPTION = """Create a new Weights & Biases Report and add text and HTML-rendered charts. Useful to save/document analysis and other findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases. 

Always provide the returned report link to the user.

<plots_html_usage_guide>
- If the analsis has generated plots then they can be logged to a Weights & Biases report via converting them to html.
- All charts should be properly rendered in raw HTML, do not use any placeholders for any chart, render everything.
- All charts should be beautiful, tasteful and well proportioned.
- Plot html code should use SVG chart elements that should render properly in any modern browser.
- Include interactive hover effects where it makes sense.
- If the analysis contains multiple charts, break up the html into one section of html per chart.
- Ensure that the axis labels are properly set and aligned for each chart.
- Always use valid markdown for the report text.
</plots_html_usage_guide>

Args:
    entity_name: str, The W&B entity (team or username) - required
    project_name: str, The W&B project name - required
    title: str, Title of the W&B Report - required
    description: str, Optional description of the W&B Report
    markdown_report_text: str, beuatifully formatted markdown text for the report body
    plots_html: str, Optional dict of plot name and html string of any charts created as part of an analysis

Returns:
    str, The url to the report

Example:
    ```python
    # Create a simple report
    report = create_report(
        entity_name="my-team",
        project_name="my-project",
        title="Model Analysis Report",
        description="Analysis of our latest model performance",
        markdown_report_text='''
            # Model Analysis Report
            [TOC]
            ## Performance Summary
            Our model achieved 95% accuracy on the test set.
            ### Key Metrics
            Precision: 0.92
            Recall: 0.89
        '''
    )
    ```
"""
