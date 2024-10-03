from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from ragagent.utils.nodes import call_model, should_continue, tool_node
from ragagent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

class createWorkflow:
    workflow = StateGraph(AgentState, config_schema=GraphConfig)

    def __init__(self) -> None:
        # workflow = StateGraph(AgentState, config_schema=GraphConfig)
        self.workflow.add_node("agent", call_model)
        self.workflow.add_node("action", tool_node)
        self.workflow.set_entry_point("agent")
        self.workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )
        self.workflow.add_edge("action", "agent")
    def getGraph(self):
        return self.workflow