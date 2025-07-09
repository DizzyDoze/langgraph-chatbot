from typing import Annotated, Any
from typing_extensions import TypedDict
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

# Configure environment
# set OPENAI_API_KEY and TAVILY_API_KEY in current env

# Define the state (same as previous tutorials)
class State(TypedDict):
    """
    State defines the structure of our chatbot as a state machine.
    Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """
    messages: Annotated[list, add_messages]

# Initialize the graph builder
graph_builder = StateGraph(State)

# Initialize the LLM
llm = init_chat_model("openai:gpt-4.1")

# Define the search tool
tool = TavilySearch(max_results=2)
tools = [tool]

# Bind tools to the LLM
# This tells the LLM which tools it can call and the correct JSON format to use
llm_with_tools = llm.bind_tools(tools)

# Add a chatbot node with tools
def chatbot(state: State):
    """
    Enhanced chatbot function that can use tools.
    The LLM will decide whether to use tools based on the user's query.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add the chatbot node
graph_builder.add_node("chatbot", chatbot)

# Add a tool node
# Use the prebuilt ToolNode which handles tool execution
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges
# The chatbot will decide whether to use tools or end the conversation
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The conditional mapping tells the graph how to interpret the condition's outputs
    {"tools": "tools", END: END},
)

# Add edges
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# 1. Create a MemorySaver checkpointer
# This is an in-memory checkpointer. In production, you would use SqliteSaver or PostgresSaver
memory = MemorySaver()

# 2. Compile the graph with the checkpointer
# This will checkpoint the State as the graph works through each node
graph = graph_builder.compile(checkpointer=memory)

# Visualize the graph
def display_graph():
    """Save the graph visualization as a PNG file."""
    try:
        graph_data = graph.get_graph().draw_mermaid_png()
        with open("chatbot_with_memory_graph.png", "wb") as f:
            f.write(graph_data)
        print("Graph saved as 'chatbot_with_memory_graph.png'")
    except Exception as e:
        print(f"Failed to save graph: {e}")
        print("Graph structure (text representation):")
        print(graph.get_graph().draw_ascii())

display_graph()

# Main chat loop with memory
if __name__ == "__main__":
    print("Chatbot with Memory")
    config = {"configurable": {"thread_id": "1"}}


    # First 
    user_input = "Hi there! My name is Will."
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # Second
    user_input = "Remember my name?"
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # Update config to different thread id
    # The only difference is we change the `thread_id` here to "2" instead of "1"
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "2"}},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


    snapshot1 = graph.get_state({"configurable": {"thread_id": "1"}})
    snapshot2 = graph.get_state({"configurable": {"thread_id": "2"}})
    # print(snapshot1, snapshot2)