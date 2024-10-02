from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from ragagent.utils.nodes import call_model, should_continue, tool_node
from ragagent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]


workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)
workflow.add_edge("action", "agent")
graph = workflow.compile()