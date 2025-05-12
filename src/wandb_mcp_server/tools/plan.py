WRITE_QUERY_PLAN_TOOL_DESCRIPTION = """Create a plan for how to answer the user's question using the available Weights & Biases data query tools.

<purpose_of_the_plan>
Weights & Biases stores an incredible amount of metrics, metadata and traces and sometime users make \
under-speficied queries. 

By writing a plan and working with the user to get any additional information, you can help to the \
correct data to answer their question.

An effective plan will also help manage the available context window of the LLM that is using these tools.
</purpose_of_the_plan>


<methods_to_improve_your_query>

<find_the_correct_column_names>
Identifying the correct column names to use is a key part of writing a good query. Some useful tactics \
include:
- querying recent metadata to get the column names. 
- querying a small number ofrecent runs or traces to get the column names.

</find_the_correct_column_names>


<filters_to_improve_your_query>
Filtering on an attribute that might encapsulate a subset of the total data in the project can \
help you find the most relevant data. Consider if the user has provided something like a run id, an \
evaluation name, a model name etc. 

Find useful values from the users query to initially filter on, these could include:
- Run ids
- Evaluation names
- Model names
- Dataset names
- Run statuses
- Run tags
- Artifact names

The above is only a subset of examples, but if nothing like the above is provided, consider asking \
the user for more information.

</filters_to_improve_your_query>

</methods_to_improve_your_query>
"""


def write_query_plan(plan: str):
    return f"This is the plan for how to efficiently and effectively answer the user's query:\n\n{plan}"