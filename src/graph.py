from langgraph.graph import StateGraph, END
from typing import Dict, Literal, Optional

from src.core.state import RepoExplainerState
from src.nodes import (
    initialize_repository,
    static_code_analyzer,
    documentation_parser,
    initial_high_level_summarizer,
    basic_user_query_parser,
    file_explainer_node,
    response_generator_node,
    architecture_analyzer_node # Added import
)

# Define node names as constants for clarity
NODE_INITIALIZE_REPO = "initialize_repository"
NODE_STATIC_CODE_ANALYZER = "static_code_analyzer"
NODE_DOC_PARSER = "documentation_parser"
NODE_INITIAL_SUMMARIZER = "initial_summarizer"
NODE_USER_QUERY_PARSER = "user_query_parser"
NODE_FILE_EXPLAINER = "file_explainer"
NODE_ARCHITECTURE_ANALYZER = "architecture_analyzer" # Added node name
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
    graph_builder.add_node(NODE_ARCHITECTURE_ANALYZER, architecture_analyzer_node) # Added node
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
        "file_explainer", "architecture_analyzer", "response_generator", "__end__" # Added architecture_analyzer
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
        elif intent == "explain_repository_architecture":
            print("Intent is repository architecture, routing to architecture_analyzer.")
            return NODE_ARCHITECTURE_ANALYZER # Route to the new architecture node
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
            NODE_ARCHITECTURE_ANALYZER: NODE_ARCHITECTURE_ANALYZER, # Added mapping
            NODE_RESPONSE_GENERATOR: NODE_RESPONSE_GENERATOR,
            END: END 
        }
    )

    # After file_explainer or architecture_analyzer, go to response_generator
    graph_builder.add_edge(NODE_FILE_EXPLAINER, NODE_RESPONSE_GENERATOR)
    graph_builder.add_edge(NODE_ARCHITECTURE_ANALYZER, NODE_RESPONSE_GENERATOR) # Added edge

    # After response_generator, decide whether to loop back or end
    def route_after_response_generator(state: RepoExplainerState) -> Literal["user_query_parser", "__end__"]:
        print(f"--- Router: route_after_response_generator ---")
        # If an error_message is present in the state at this point,
        # it implies that response_generator likely just processed this error.
        # To prevent an immediate loop if the error condition persists 
        # (e.g., user_query_parser failing again on the same input/state),
        # we terminate the graph.
        # A more robust solution might involve response_generator clearing the error_message
        # once handled, or nodes using retry counters for transient errors.
        if state.get("error_message"):
            print(f"Error message \"{state.get('error_message')}\" is present after response_generator. Ending graph to prevent loop.")
            return END
        
        print("Response generated. Ending graph for this query-response cycle.")
        return END

    graph_builder.add_conditional_edges(
        NODE_RESPONSE_GENERATOR,
        route_after_response_generator,
        {
            NODE_USER_QUERY_PARSER: NODE_USER_QUERY_PARSER,
            END: END
        }
    )
    
    # Compile the graph
    app = graph_builder.compile()
    return app

if __name__ == '__main__':
    import os
    import shutil
    import sys # Added to access command-line arguments

    # Determine repo_url and local_repo_path based on input
    # Priority:
    # 1. Command-line argument (sys.argv[1])
    # 2. REPO_URL environment variable
    # 3. Default to local test repo if neither is provided (for direct script runs without args)

    input_repo_url: Optional[str] = None
    input_local_repo_path: Optional[str] = None
    user_query_from_args: str = "Tell me about this repository." # Default query

    if len(sys.argv) > 1:
        # If the first argument is a URL (simple check, can be improved)
        if sys.argv[1].startswith("http://") or sys.argv[1].startswith("https://") or sys.argv[1].endswith(".git"):
            input_repo_url = sys.argv[1]
            print(f"Using repo_url from command-line argument: {input_repo_url}")
            if len(sys.argv) > 2: # If there's a second argument, assume it's the query
                user_query_from_args = " ".join(sys.argv[2:]) # Join all remaining args as query
        else:
            # Assume it's a local path if not a URL-like string
            input_local_repo_path = sys.argv[1]
            print(f"Using local_repo_path from command-line argument: {input_local_repo_path}")
            if len(sys.argv) > 2:
                user_query_from_args = " ".join(sys.argv[2:])
    else:
        # Fallback to environment variable if no command-line arg
        env_repo_url = os.getenv("REPO_URL")
        if env_repo_url:
            input_repo_url = env_repo_url
            print(f"Using repo_url from REPO_URL environment variable: {input_repo_url}")
        # If REPO_URL env var also not set, then we might use the dummy repo
        # or a default local path if specified by another env var like LOCAL_REPO_PATH

    # Setup for local test repo if no URL or specific local path is provided
    test_app_repo_dir = "./temp_app_test_repo" # Default test directory
    if not input_repo_url and not input_local_repo_path:
        print(f"No repo_url or local_repo_path provided via args/env. Using default test repo: {test_app_repo_dir}")
        input_local_repo_path = test_app_repo_dir
        # Create a dummy repo for testing the full graph if using the default
        if os.path.exists(test_app_repo_dir): # Clean up from previous runs
            shutil.rmtree(test_app_repo_dir)
        os.makedirs(os.path.join(test_app_repo_dir, "src"), exist_ok=True)
        with open(os.path.join(test_app_repo_dir, "README.md"), "w") as f:
            f.write("# Test App Repo\nThis is a test application for the explainer agent.")
        with open(os.path.join(test_app_repo_dir, "src", "main.py"), "w") as f:
            f.write("def main_function():\n    print('Hello from main')\n\nclass MainClass:\n    pass")
        with open(os.path.join(test_app_repo_dir, "src", "utils.py"), "w") as f:
            f.write("def helper():\n    return 'helper value'")
    elif input_local_repo_path and not os.path.isdir(input_local_repo_path):
        print(f"Error: Provided local_repo_path '{input_local_repo_path}' does not exist or is not a directory.")
        sys.exit(1)


    # --- Test Scenario 1: Initial run with a query for overview ---
    print("\n--- SCENARIO 1: Initial Run - Overview Query ---")
    app = build_graph()
    initial_input_overview: RepoExplainerState = {
        "repo_url": input_repo_url,
        "local_repo_path": input_local_repo_path,
        "user_query": user_query_from_args, # Use query from args or default
        "explanation_history": []
    }
    
    # Stream events to see the flow
    print("\nInvoking graph for overview...")
    _final_state_after_run1 = None
    for _event_state in app.stream(initial_input_overview, stream_mode="values"):
        _final_state_after_run1 = _event_state # Capture the full state after each step

    # After the stream, _final_state_after_run1 is the complete state at the end of the graph.
    # Check if 'generated_explanation' is present and non-empty in the final state.
    final_state_overview = None # Initialize to ensure it's None if checks fail
    if _final_state_after_run1:
        explanation_run1 = _final_state_after_run1.get("generated_explanation")
        print(f"DEBUG Scenario 1: 'generated_explanation' from final state is: {repr(explanation_run1)}")
        if explanation_run1: # Check if it's a non-empty string
            print(f"\n--- Output from Response Generator (Overview) ---")
            print(explanation_run1)
            final_state_overview = _final_state_after_run1 # Assign the full state for Scenario 2
        else:
            print(f"Scenario 1: 'generated_explanation' is present but falsey (None or empty). Final state keys: {list(_final_state_after_run1.keys())}")
    else:
        print("Scenario 1: Did not complete successfully (final state is None).")

    # --- Test Scenario 2 has been removed to only process the command-line query ---

    # Cleanup for the default test repo if it was created
    if input_local_repo_path == test_app_repo_dir and os.path.exists(test_app_repo_dir):
        print(f"Cleaning up default test repo: {test_app_repo_dir}")
        shutil.rmtree(test_app_repo_dir)
    
    # General cleanup for cloned repositories (if any were cloned by initialize_repository_node)
    # This might be better handled by the initialize_repository_node itself or a dedicated cleanup function
    # if the script is intended to be used as a library. For a CLI tool, this cleanup is reasonable.
    cloned_repos_dir = "./cloned_repositories" 
    if os.path.exists(cloned_repos_dir):
        # Add a check to ensure we are not deleting something unintended if the script is run from a different context
        # For now, assuming this is run from the project root where "./cloned_repositories" is expected.
        print(f"Cleaning up cloned repositories directory: {cloned_repos_dir}")
        # shutil.rmtree(cloned_repos_dir) # Temporarily commenting out to avoid accidental deletion during testing
        # print(f"Note: Automatic cleanup of {cloned_repos_dir} is currently commented out in the script.")


    print("\nGraph processing for the provided query completed.")
