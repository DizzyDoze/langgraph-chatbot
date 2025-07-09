from typing import Annotated
from typing_extensions import TypedDict
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

# Configure environment
# set OPENAI_API_KEY and TAVILY_API_KEY in current env

# Step 1: Define the state (same as basic chatbot)
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

# 2 Initialize the LLM
llm = init_chat_model("openai:gpt-4.1")

# 3 Define the search tool
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]

# 4 Bind tools to the LLM
# This tells the LLM which tools it can call and the correct JSON format to use
llm_with_tools = llm.bind_tools(tools)

# 5 Add a chatbot node with tools
def chatbot_with_tools(state: State):
    """
    Enhanced chatbot function that can use tools.
    The LLM will decide whether to use tools based on the user's query.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add the chatbot node
graph_builder.add_node("chatbot_with_tools", chatbot_with_tools)

# 6 Add a tool node
# Use the prebuilt ToolNode which handles tool execution
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 7 Add conditional edges
# The chatbot will decide whether to use tools or end the conversation
graph_builder.add_conditional_edges(
    "chatbot_with_tools",
    tools_condition,
    # The conditional mapping tells the graph how to interpret the condition's outputs
    # It defaults to the identity function, but you can customize it
    {"tools": "tools", END: END},
)

# 8 Add edges
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot_with_tools")
graph_builder.add_edge(START, "chatbot_with_tools")

# 9 Compile the graph
graph = graph_builder.compile()

# Visualize the graph
def display_graph():
    """Save the graph visualization as a PNG file."""
    try:
        graph_data = graph.get_graph().draw_mermaid_png()
        with open("chatbot_with_tools_graph.png", "wb") as f:
            f.write(graph_data)
        print("Graph saved as 'chatbot_with_tools_graph.png'")
    except Exception as e:
        print(f"Failed to save graph: {e}")
        print("Graph structure (text representation):")
        print(graph.get_graph().draw_ascii())

display_graph()

# 11 Run the chatbot with tools
def stream_graph_updates(user_input: str):
    """
    Stream graph updates for the given user input.
    The chatbot will use tools if needed to provide accurate responses.
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        if event is None:
            continue
        for value in event.values():
            if value is None:
                continue
            if "messages" in value and value["messages"]:
                print("Assistant:", value["messages"][-1].content)

# Main chat loop
if __name__ == "__main__":
    print("Chatbot with Tool")
    print("This chatbot can search the web for current information!")
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break 