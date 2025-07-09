from typing import Annotated
from typing_extensions import TypedDict
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

# Configure environment
# set OPENAI_API_KEY in current env

# 1. Create a StateGraph
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

# 2. Initialize the LLM
llm = init_chat_model("openai:gpt-4.1")

# 3. Add a chatbot node
def chatbot(state: State):
    """
    Notice how the chatbot node function takes the current State as input 
    and returns a dictionary containing an updated messages list under the key "messages". 
    This is the basic pattern for all LangGraph node functions.
    """
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used
graph_builder.add_node("chatbot", chatbot)

# 4. Add an entry point
# Tell the graph where to start its work each time it is run
graph_builder.add_edge(START, "chatbot")

# 5. Add an exit point
# Tell the graph where to finish execution (terminate after chatbot node)
graph_builder.add_edge("chatbot", END)

# 6. Compile the graph
graph = graph_builder.compile()

# Visualize the graph
def display_graph():
    """Save the graph visualization as a PNG file."""
    try:
        graph_data = graph.get_graph().draw_mermaid_png()
        with open("basic_chatbot_graph.png", "wb") as f:
            f.write(graph_data)
        print("Graph saved as 'basic_chatbot_graph.png'")
    except Exception as e:
        print(f"Failed to save graph: {e}")
        print("Graph structure (text representation):")
        print(graph.get_graph().draw_ascii())

display_graph()

# 8. Run the chatbot
def stream_graph_updates(user_input: str):
    """Stream graph updates for the given user input."""
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
    print("Basic Chatbot")
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