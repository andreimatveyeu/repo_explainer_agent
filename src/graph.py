from langgraph.graph import StateGraph, END
from typing import Dict, Literal

from src.core.state import RepoExplainerState
from src.nodes import (
    initialize_repository,
    static_code_analyzer,
    documentation_parser,
    initial_high_level_summarizer,
    basic_user_query_parser,
    file_explainer_node,
    response_generator_node
)

# Define node names as constants for clarity
NODE_INITIALIZE_REPO = "initialize_repository"
NODE_STATIC_CODE_ANALYZER = "static_code_analyzer"
NODE_DOC_PARSER = "documentation_parser"
NODE_INITIAL_SUMMARIZER = "initial_summarizer"
NODE_USER_QUERY_PARSER = "user_query_parser"
NODE_FILE_EXPLAINER = "file_explainer"
NODE_RESPONSE_GENERATOR = "response_generator"
# NODE_ERROR_HANDLER = "error_handler" # For future error handling logic

def build_graph() -> StateGraph:
    """
    Builds and configures the LangGraph for the Repository Explainer Agent.
    """
    graph_builder = StateGraph(RepoExplainerState)

    # Add nodes to the graph
    graph_builder.add_node(NODE_INITIALIZE_REPO, initialize_repository)
    graph_builder.add_node(NODE_STATIC_CODE_ANALYZER, static_code_analyzer)
    graph_builder.add_node(NODE_DOC_PARSER, documentation_parser)
    graph_builder.add_node(NODE_INITIAL_SUMMARIZER, initial_high_level_summarizer)
    graph_builder.add_node(NODE_USER_QUERY_PARSER, basic_user_query_parser)
    graph_builder.add_node(NODE_FILE_EXPLAINER, file_explainer_node)
    graph_builder.add_node(NODE_RESPONSE_GENERATOR, response_generator_node)
    # graph_builder.add_node(NODE_ERROR_HANDLER, error_handler_node) # Future

    # Define the entry point for the graph
    graph_builder.set_entry_point(NODE_INITIALIZE_REPO)

    # Define edges for the initial data ingestion flow
    graph_builder.add_edge(NODE_INITIALIZE_REPO, NODE_STATIC_CODE_ANALYZER)
    graph_builder.add_edge(NODE_STATIC_CODE_ANALYZER, NODE_DOC_PARSER)
    graph_builder.add_edge(NODE_DOC_PARSER, NODE_INITIAL_SUMMARIZER)
    
    # After initial summarization, wait for user query (or proceed to query parser if query already in state)
    graph_builder.add_edge(NODE_INITIAL_SUMMARIZER, NODE_USER_QUERY_PARSER)

    # Conditional routing from user_query_parser
    def route_after_query_parser(state: RepoExplainerState) -> Literal[
        "file_explainer", "response_generator", "__end__" # Added __end__ for robustness
    ]:
        print(f"--- Router: route_after_query_parser ---")
        if state.get("error_message"):
            print(f"Error detected, routing to response_generator to output error.")
            return NODE_RESPONSE_GENERATOR # Let response_generator handle displaying the error
        
        intent = state.get("parsed_query_intent")
        print(f"Parsed intent: {intent}")
        
        if intent == "explain_file":
            # Check if a file is actually targeted
            if state.get("current_focus_path") or state.get("target_entity_ids"):
                print("Routing to file_explainer.")
                return NODE_FILE_EXPLAINER
            else:
                print("Intent was explain_file, but no file target. Routing to response_generator for clarification/error.")
                # Update state to reflect this issue for response_generator
                state["error_message"] = "Intent was to explain a file, but no specific file was identified."
                return NODE_RESPONSE_GENERATOR
        elif intent == "explain_repository_overview":
            print("Intent is repository overview, routing to response_generator.")
            # overall_summary should already be in state from initial_summarizer
            # response_generator will pick it up.
            return NODE_RESPONSE_GENERATOR
        elif intent == "find_code_entity":
            # For Phase 1, we don't have a dedicated node for this yet.
            # We can route to response_generator which might state this capability is upcoming,
            # or try a generic explanation if target_entity_ids were populated.
            # Let's assume for now it means we should try to explain the file of the found entity.
            if state.get("target_entity_ids") and state.get("current_focus_path"):
                 print("Intent find_code_entity, routing to file_explainer for the entity's file.")
                 return NODE_FILE_EXPLAINER
            print("Intent find_code_entity, but no clear target or capability. Routing to response_generator.")
            state["generated_explanation"] = "I can find code entities, but explaining them in detail beyond their file context is a future enhancement. For now, I'll describe the file they are in if identified."
            return NODE_RESPONSE_GENERATOR
        else: # unclear_intent or other fallbacks
            print("Intent unclear or not directly actionable by a specialized node, routing to response_generator.")
            if not state.get("generated_explanation"): # If no other node set an explanation
                 state["generated_explanation"] = "I'm not sure how to handle that specific request yet, or the intent was unclear."
            return NODE_RESPONSE_GENERATOR

    graph_builder.add_conditional_edges(
        NODE_USER_QUERY_PARSER,
        route_after_query_parser,
        {
            NODE_FILE_EXPLAINER: NODE_FILE_EXPLAINER,
            NODE_RESPONSE_GENERATOR: NODE_RESPONSE_GENERATOR,
            END: END # Should not happen with current router logic but good for completeness
        }
    )

    # After file_explainer, go to response_generator
    graph_builder.add_edge(NODE_FILE_EXPLAINER, NODE_RESPONSE_GENERATOR)

    # After response_generator, loop back to user_query_parser to await next query
    # This creates the conversational loop.
    # For a single-shot run (like initial summarization without a query yet),
    # the graph might end if no user_query is subsequently injected.
    # Or, the user_query_parser could be the point where the agent "waits".
    # For now, let's make it loop back.
    graph_builder.add_edge(NODE_RESPONSE_GENERATOR, NODE_USER_QUERY_PARSER)
    
    # Compile the graph
    app = graph_builder.compile()
    return app

if __name__ == '__main__':
    import os
    import shutil

    # Create a dummy repo for testing the full graph
    test_app_repo_dir = "./temp_app_test_repo"
    if os.path.exists(test_app_repo_dir): # Clean up from previous runs
        shutil.rmtree(test_app_repo_dir)
    os.makedirs(os.path.join(test_app_repo_dir, "src"), exist_ok=True)

    with open(os.path.join(test_app_repo_dir, "README.md"), "w") as f:
        f.write("# Test App Repo\nThis is a test application for the explainer agent.")
    with open(os.path.join(test_app_repo_dir, "src", "main.py"), "w") as f:
        f.write("def main_function():\n    print('Hello from main')\n\nclass MainClass:\n    pass")
    with open(os.path.join(test_app_repo_dir, "src", "utils.py"), "w") as f:
        f.write("def helper():\n    return 'helper value'")

    # --- Test Scenario 1: Initial run with a query for overview ---
    print("\n--- SCENARIO 1: Initial Run - Overview Query ---")
    app = build_graph()
    initial_input_overview: RepoExplainerState = {
        "repo_url": None, # Specify local path instead
        "local_repo_path": test_app_repo_dir,
        "user_query": "Tell me about this repository.",
        "explanation_history": []
    }
    
    # Stream events to see the flow
    print("\nInvoking graph for overview...")
    final_state_overview = None
    for event in app.stream(initial_input_overview, stream_mode="values"):
        # print(f"\nEvent: {event}")
        # The event here is the full state after each node execution
        # We are interested in the final state or specific parts of it.
        # The 'values' stream_mode gives the full state dict.
        # 'updates' would give only the changed keys.
        current_state_snapshot = event 
        # final_state_overview = event # Keep updating, last one is final
        # Let's print the output of response_generator when it runs
        if NODE_RESPONSE_GENERATOR in current_state_snapshot: # Check if the key for the node output exists
            print(f"\n--- Output from Response Generator (Overview) ---")
            print(current_state_snapshot[NODE_RESPONSE_GENERATOR].get("generated_explanation"))
            final_state_overview = current_state_snapshot[NODE_RESPONSE_GENERATOR]


    # --- Test Scenario 2: Follow-up query to explain a file ---
    print("\n\n--- SCENARIO 2: Follow-up Query - Explain File ---")
    if final_state_overview: # Ensure previous state is available
        # Prepare state for the next query, carrying over relevant parts
        # The graph loops back to user_query_parser, so it expects a 'user_query'
        # and uses existing KB, summary, history.
        
        # The 'final_state_overview' is the output of the response_generator node,
        # which is a dict containing the full updated state.
        
        input_for_file_query: RepoExplainerState = {
            **final_state_overview, # Carry over all previous state
            "user_query": f"Explain the file src/main.py", 
            # No need to reset explanation_history, it should append
        }
        
        print("\nInvoking graph for file explanation...")
        final_state_file_explain = None
        for event in app.stream(input_for_file_query, stream_mode="values"):
            current_state_snapshot = event
            if NODE_RESPONSE_GENERATOR in current_state_snapshot:
                print(f"\n--- Output from Response Generator (File Explain) ---")
                print(current_state_snapshot[NODE_RESPONSE_GENERATOR].get("generated_explanation"))
                final_state_file_explain = current_state_snapshot[NODE_RESPONSE_GENERATOR]
    else:
        print("Skipping Scenario 2 as Scenario 1 did not complete successfully.")

    # Cleanup
    if os.path.exists(test_app_repo_dir):
        shutil.rmtree(test_app_repo_dir)
    cloned_repos_dir = "./cloned_repositories" # From initialize_repository_node
    if os.path.exists(cloned_repos_dir):
         shutil.rmtree(cloned_repos_dir)

    print("\nGraph definition and basic test runs completed.")
