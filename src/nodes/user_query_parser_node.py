import json
from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState
from src.core.models import AbstractCodeKnowledgeBase
from src.utils.llm_utils import call_llm_for_structured_output, call_llm_for_summary

# Define a simple set of intents for the basic parser
# In a more advanced system, these could be more dynamic or complex.
SUPPORTED_INTENTS = [
    "explain_file",
    "explain_repository_overview",  # General question about the repo
    "explain_repository_architecture", # Detailed question about repo structure, components, architecture
    "find_code_entity",  # E.g. "where is function X defined?"
    "unclear_intent"  # Fallback
]

def basic_user_query_parser(state: RepoExplainerState) -> RepoExplainerState:
    """
    Parses the user_query to understand intent and identify target entities.
    This is a basic version for Phase 1.
    Updates `parsed_query_intent`, `target_entity_ids` (or `current_focus_path` for files),
    and potentially `error_message` in the state.
    """
    print("--- Running Node: user_query_parser (Basic) ---")
    user_query: Optional[str] = state.get("user_query")
    overall_summary: Optional[str] = state.get("overall_summary")
    code_kb: Optional[AbstractCodeKnowledgeBase] = state.get("abstract_code_kb")
    
    updated_state: Dict[str, Any] = {
        "parsed_query_intent": "unclear_intent", # Default
        "target_entity_ids": [],
        "current_focus_path": None,
        "error_message": None
    }

    if not user_query:
        message = "User query is missing. Cannot parse."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        return {**state, **updated_state}

    # Prepare context for the LLM
    context_chunks: List[str] = []
    if overall_summary:
        context_chunks.append(f"Overall Repository Summary:\n{overall_summary}")
    
    if code_kb:
        context_chunks.append("Available files in the repository:")
        # List a few top-level files as context
        file_list_preview = "\n".join(
            [f"- {fp}" for fp in code_kb.file_paths[:10]] # Show up to 10 files
        )
        if len(code_kb.file_paths) > 10:
            file_list_preview += "\n... and more files."
        context_chunks.append(file_list_preview)
        
        # List some key entities if available
        key_entities_preview_parts = []
        count = 0
        for entity_id, entity in code_kb.entities.items():
            if entity.entity_type in ["class", "function", "method"] and count < 5:
                 key_entities_preview_parts.append(f"- {entity.entity_type} '{entity.name}' in '{entity.filepath}'")
                 count += 1
            if count >= 5:
                break
        if key_entities_preview_parts:
            context_chunks.append("\nSome known code entities:")
            context_chunks.append("\n".join(key_entities_preview_parts))


    prompt = (
        f"You are an AI assistant helping to understand user queries about a software repository. "
        f"The user's query is: \"{user_query}\"\n\n"
        f"Based on this query and the provided context about the repository, determine the user's primary intent. "
        f"Supported intents are: {', '.join(SUPPORTED_INTENTS)}.\n"
        f"- Use 'explain_repository_overview' for general, high-level questions about the repository (e.g., 'What is this repo about?').\n"
        f"- Use 'explain_repository_architecture' for questions asking for a more detailed explanation of the repository's structure, components, architecture, or how parts work together (e.g., 'Explain the architecture', 'How are the components organized?').\n"
        f"- Use 'explain_file' if the query is specifically about a single file (e.g., 'Explain src/main.py'). Identify the file path.\n"
        f"- Use 'find_code_entity' if the query is about locating a specific function or class (e.g., 'Where is MyClass defined?'). Identify its name.\n"
        f"If the query is about a specific file, identify the file path. "
        f"If it's about a specific function or class, identify its name and, if possible, the file it might be in."
    )
    
    output_format_desc = (
        "Return a JSON object with the following keys:\n"
        "- 'intent': (string) One of the supported intents.\n"
        "- 'target_file': (string, optional) The path of the file the user is asking about.\n"
        "- 'target_entity_name': (string, optional) The name of the function/class the user is asking about.\n"
        "- 'confidence': (float) Your confidence in this interpretation (0.0 to 1.0)."
    )

    # Using call_llm_for_structured_output which simulates returning a dict
    # llm_response_dict = call_llm_for_structured_output(
    #     prompt, 
    #     output_format_description=output_format_desc, 
    #     context_chunks=context_chunks
    # )
    
    # For Phase 1, let's use the simpler call_llm_for_summary and parse its string output
    # as the structured output simulation might be too specific for the generic placeholder.
    # We'll simulate the JSON string response that the simpler LLM might give.
    # This also allows testing the JSON parsing logic here.
    
    # Construct a more detailed prompt for the simpler LLM to guide it to produce JSON-like string
    detailed_prompt_for_json_string = prompt + \
        f"\n\nPlease format your response as a single-line JSON string with keys: " \
        f"'intent' (one of {SUPPORTED_INTENTS}), " \
        f"'target_file' (e.g., 'src/main.py', or null if not applicable), " \
        f"'target_entity_name' (e.g., 'MyClass', or null if not applicable), " \
        f"and 'confidence' (a float between 0.0 and 1.0)."

    llm_response_str = call_llm_for_summary(
        detailed_prompt_for_json_string,
        context_chunks=context_chunks,
        max_tokens=500, 
        temperature=0.1
    )

    if llm_response_str and llm_response_str.strip():
        cleaned_llm_response_str = llm_response_str.strip()
        
        # Attempt to remove common markdown code fences
        if cleaned_llm_response_str.startswith("```json") and cleaned_llm_response_str.endswith("```"):
            cleaned_llm_response_str = cleaned_llm_response_str[len("```json"):-len("```")].strip()
        elif cleaned_llm_response_str.startswith("```") and cleaned_llm_response_str.endswith("```"):
            cleaned_llm_response_str = cleaned_llm_response_str[len("```"):-len("```")].strip()

        try:
            parsed_llm_output = json.loads(cleaned_llm_response_str)
            
            intent = parsed_llm_output.get("intent", "unclear_intent")
            if intent not in SUPPORTED_INTENTS:
                print(f"Warning: LLM returned unsupported intent '{intent}'. Defaulting to 'unclear_intent'.")
                intent = "unclear_intent"
            updated_state["parsed_query_intent"] = intent

            target_file = parsed_llm_output.get("target_file")
            target_entity_name = parsed_llm_output.get("target_entity_name")

            if target_file:
                # In a real scenario, validate if target_file exists in code_kb.file_paths
                # For now, just use it if provided.
                updated_state["current_focus_path"] = target_file
                # If a file is targeted, its ID in KB is often filepath::filepath
                if code_kb and f"{target_file}::{target_file}" in code_kb.entities:
                     updated_state["target_entity_ids"] = [f"{target_file}::{target_file}"]


            if target_entity_name and code_kb:
                # Attempt to find this entity in the KB. This is a simple name match.
                # A more robust search would consider context (e.g., current_focus_path).
                found_entities = code_kb.find_entities(name=target_entity_name)
                if found_entities:
                    updated_state["target_entity_ids"] = [entity.id for entity in found_entities]
                    # If multiple matches, might need disambiguation in a later step or refine here.
                    # For now, take all matches. If a target_file was also identified, could filter by that.
                    if target_file:
                        updated_state["target_entity_ids"] = [
                            eid for eid in updated_state["target_entity_ids"] if code_kb.get_entity(eid).filepath == target_file
                        ]
                    if updated_state["target_entity_ids"] and not updated_state.get("current_focus_path"):
                        # If we found specific entities, set focus path to the first one's file
                        updated_state["current_focus_path"] = code_kb.get_entity(updated_state["target_entity_ids"][0]).filepath


            print(f"Parsed query: Intent='{intent}', File='{target_file}', Entity='{target_entity_name}', Target IDs='{updated_state['target_entity_ids']}'")

        except json.JSONDecodeError as e:
            message = f"Failed to parse LLM response for query understanding. Error: {e}. Original Response: '{llm_response_str}'. Cleaned Response: '{cleaned_llm_response_str}'"
            print(f"Error: {message}")
            updated_state["error_message"] = message
        except Exception as e:
            message = f"Unexpected error processing LLM response: {e}"
            print(f"Error: {message}")
            updated_state["error_message"] = message
    else:
        message = "LLM did not return a response for query understanding."
        print(f"Error: {message}")
        updated_state["error_message"] = message

    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage
    import os
    from src.core.models import CodeEntity # For mock_kb

    # Mock CodeKnowledgeBase
    mock_kb_instance = AbstractCodeKnowledgeBase()
    mock_kb_instance.file_paths = ["src/main.py", "src/utils.py", "README.md"]
    mock_kb_instance.add_entity(CodeEntity(id="src/main.py::src/main.py", entity_type="file", name="main.py", qualified_name="src/main.py", language="python", filepath="src/main.py", start_line=1, end_line=10))
    mock_kb_instance.add_entity(CodeEntity(id="src/main.py::MyClass", entity_type="class", name="MyClass", qualified_name="MyClass", language="python", filepath="src/main.py", start_line=1, end_line=10))
    mock_kb_instance.add_entity(CodeEntity(id="src/utils.py::helper_func", entity_type="function", name="helper_func", qualified_name="helper_func", language="python", filepath="src/utils.py", start_line=1, end_line=5))


    test_queries = [
        "Tell me about this repository.",
        "Explain the file src/main.py",
        "What does MyClass do?",
        "Can you find the function helper_func?",
        "Gibberish query here."
    ]

    # Simulate that initial_high_level_summarizer has run
    simulated_overall_summary = "This is a Python project with main.py and utils.py."

    for query in test_queries:
        print(f"\n--- Testing query: \"{query}\" ---")
        initial_state: RepoExplainerState = {
            "user_query": query,
            "overall_summary": simulated_overall_summary,
            "abstract_code_kb": mock_kb_instance,
            "explanation_history": [] # Initialize
        }
        result_state = basic_user_query_parser(initial_state)
        
        print(f"Intent: {result_state.get('parsed_query_intent')}")
        print(f"Target File (current_focus_path): {result_state.get('current_focus_path')}")
        print(f"Target Entity IDs: {result_state.get('target_entity_ids')}")
        print(f"Error: {result_state.get('error_message')}")

        # Basic assertions based on simulated LLM behavior
        if "explain the file" in query.lower():
            assert result_state.get("parsed_query_intent") == "explain_file"
            assert result_state.get("current_focus_path") is not None
        elif "myclass" in query.lower():
             # The placeholder LLM might not be smart enough to always get this right without better context/prompting
             # For now, we rely on its simulated output.
            pass


    print("\nuser_query_parser (basic) node tests completed.")
