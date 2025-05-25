from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState
from src.core.models import AbstractCodeKnowledgeBase, CodeEntity
from src.utils.llm_utils import call_llm_for_summary

MAX_ENTITIES_FOR_FILE_EXPLANATION = 15 # Limit how many entities from a file we detail in the prompt
MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT = 500 # Limit length of raw code snippets in prompt

def format_file_entities_for_llm(
    file_path: str, 
    file_entities: List[CodeEntity],
    code_kb: AbstractCodeKnowledgeBase # To fetch children if needed
) -> List[str]:
    """
    Formats CodeEntity objects from a specific file for an LLM prompt.
    """
    if not file_entities:
        return [f"No specific code structures (classes, functions, etc.) were parsed for the file: {file_path}."]

    context_chunks = []
    
    file_level_entity = None
    top_level_structural_entities = [] # e.g. classes, functions directly in the file

    for entity in file_entities:
        if entity.entity_type == "file" and entity.filepath == file_path:
            file_level_entity = entity
        elif entity.parent_id and code_kb.get_entity(entity.parent_id) and \
             code_kb.get_entity(entity.parent_id).entity_type == "file" and \
             code_kb.get_entity(entity.parent_id).filepath == file_path:
            top_level_structural_entities.append(entity)
    
    if file_level_entity:
        context_chunks.append(f"Explaining file: {file_path}")
        if file_level_entity.docstring:
            context_chunks.append(f"File-level Docstring:\n{file_level_entity.docstring[:MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT]}")
        if file_level_entity.summary: # If a summary was pre-generated for the file entity
             context_chunks.append(f"Pre-existing File Summary:\n{file_level_entity.summary}")
    else: # Should not happen if parser creates a file entity
        context_chunks.append(f"Details for file: {file_path}")

    if not top_level_structural_entities:
        context_chunks.append("The file does not seem to contain top-level classes or functions, or they were not parsed in detail.")
        # We might still have the raw_text_snippet of the file_level_entity to use
        if file_level_entity and file_level_entity.raw_text_snippet:
             context_chunks.append(f"\nFull File Content (first {MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT} chars):\n{file_level_entity.raw_text_snippet[:MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT]}")
        return context_chunks

    context_chunks.append("\nKey structures within this file:")
    for i, entity in enumerate(top_level_structural_entities):
        if i >= MAX_ENTITIES_FOR_FILE_EXPLANATION:
            context_chunks.append(f"... and {len(top_level_structural_entities) - i} more top-level structures.")
            break
        
        chunk = f"\n--- {entity.entity_type.capitalize()} Name: {entity.name} ---"
        if entity.qualified_name and entity.qualified_name != entity.name:
            chunk += f"\nQualified Name: {entity.qualified_name}"
        chunk += f"\nLines: {entity.start_line} - {entity.end_line}"
        if entity.docstring:
            chunk += f"\nDocstring:\n{entity.docstring[:MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT]}"
            if len(entity.docstring) > MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT:
                chunk += "..."
        if entity.summary: # If entity itself has a pre-summary
            chunk += f"\nPre-existing Entity Summary: {entity.summary}"
        
        # Add metadata like parameters for functions/methods
        if entity.entity_type in ["function", "method"] and entity.metadata:
            params = entity.metadata.get("parameters")
            if params:
                chunk += f"\nParameters: {', '.join(params)}"
            ret_type = entity.metadata.get("return_type_annotation")
            if ret_type:
                 chunk += f"\nReturns (annotation): {ret_type}"
        
        if entity.raw_text_snippet: # Add snippet of code if available and not too long
            chunk += f"\nCode Snippet:\n{entity.raw_text_snippet[:MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT]}"
            if len(entity.raw_text_snippet) > MAX_RAW_SNIPPET_LENGTH_FOR_PROMPT:
                chunk += "..."
        context_chunks.append(chunk)
        
    return context_chunks


def file_explainer_node(state: RepoExplainerState) -> RepoExplainerState:
    """
    Generates an explanation for a specific file identified by `current_focus_path`
    or the file associated with the first `target_entity_ids`.
    Uses `abstract_code_kb` and an LLM.
    Updates `generated_explanation` in the state.
    """
    print("--- Running Node: file_explainer_node ---")
    current_focus_path: Optional[str] = state.get("current_focus_path")
    target_entity_ids: Optional[List[str]] = state.get("target_entity_ids")
    code_kb: Optional[AbstractCodeKnowledgeBase] = state.get("abstract_code_kb")
    
    updated_state: Dict[str, Any] = {"generated_explanation": None, "error_message": None}

    if not code_kb:
        message = "AbstractCodeKnowledgeBase not available. Cannot explain file."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        updated_state["generated_explanation"] = "Error: Code knowledge base is missing."
        return {**state, **updated_state}

    file_to_explain_path: Optional[str] = current_focus_path
    
    # If no direct focus path, try to get it from target_entity_ids (e.g., if a file entity was targeted)
    if not file_to_explain_path and target_entity_ids:
        first_target_id = target_entity_ids[0]
        target_entity = code_kb.get_entity(first_target_id)
        if target_entity and target_entity.entity_type == "file":
            file_to_explain_path = target_entity.filepath
        elif target_entity: # If it's not a file entity, use its filepath
            file_to_explain_path = target_entity.filepath


    if not file_to_explain_path:
        message = "No target file specified for explanation (current_focus_path or target_entity_ids pointing to a file)."
        print(f"Error: {message}")
        # This might not be an error if the intent wasn't file-specific.
        # The graph logic should route appropriately.
        # For now, if this node is called, it expects a file.
        updated_state["error_message"] = message
        updated_state["generated_explanation"] = "Error: No file was specified to be explained."
        return {**state, **updated_state}

    # Retrieve all entities belonging to this file from the KB
    # This includes the file entity itself, plus functions, classes, etc. within it.
    entities_in_file = code_kb.get_file_entities(file_to_explain_path)

    if not entities_in_file:
        message = f"No information found in the knowledge base for file: {file_to_explain_path}"
        print(f"Warning: {message}")
        updated_state["generated_explanation"] = f"Could not find detailed information for '{file_to_explain_path}' in the code knowledge base. It might be an empty file, a non-code file, or was not parsed."
        # updated_state["error_message"] = message # Not necessarily a critical error for the whole flow
        return {**state, **updated_state}

    context_chunks = format_file_entities_for_llm(file_to_explain_path, entities_in_file, code_kb)
    
    prompt = (
        f"You are an expert software engineering assistant. Based on the provided information "
        f"about the file '{file_to_explain_path}' (including its top-level structures like classes/functions, "
        f"their docstrings, and code snippets), generate a comprehensive explanation of this file. "
        f"Your explanation should cover:\n"
        f"1. The primary purpose or role of this file within the larger project (if inferable).\n"
        f"2. A description of each major class and/or function defined directly within the file, including its functionality.\n"
        f"3. How these components interact with each other (if apparent from the provided details).\n"
        f"4. Any key data structures or important logic highlighted in the snippets or docstrings.\n\n"
        f"Aim for a clear and informative explanation suitable for a developer trying to understand this specific file."
    )

    explanation = call_llm_for_summary(prompt, context_chunks=context_chunks, max_tokens=700) # Allow longer explanation

    if explanation:
        print(f"Generated explanation for {file_to_explain_path}: {explanation[:200]}...")
        updated_state["generated_explanation"] = explanation
    else:
        message = f"LLM failed to generate explanation for file: {file_to_explain_path}"
        print(f"Error: {message}")
        updated_state["error_message"] = message
        updated_state["generated_explanation"] = f"Could not generate an explanation for '{file_to_explain_path}' at this time."
        
    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage
    import os 
    from src.core.models import CodeEntity # For mock_kb

    # Mock CodeKnowledgeBase
    mock_kb_instance = AbstractCodeKnowledgeBase()
    
    # File 1: main.py
    main_py_path = "project/src/main.py"
    main_py_file_entity = CodeEntity(id=f"{main_py_path}::{main_py_path}", entity_type="file", name="main.py", qualified_name=main_py_path, language="python", filepath=main_py_path, start_line=1, end_line=20, docstring="This is the main application file.", raw_text_snippet="class MainApp:\n  pass\ndef run_app():\n  pass")
    main_class_entity = CodeEntity(id=f"{main_py_path}::MainApp", entity_type="class", name="MainApp", qualified_name="MainApp", language="python", filepath=main_py_path, start_line=1, end_line=2, docstring="Main application class.", parent_id=main_py_file_entity.id)
    main_func_entity = CodeEntity(id=f"{main_py_path}::run_app", entity_type="function", name="run_app", qualified_name="run_app", language="python", filepath=main_py_path, start_line=3, end_line=4, docstring="Runs the application.", parent_id=main_py_file_entity.id, metadata={"parameters":[]})
    
    mock_kb_instance.add_entity(main_py_file_entity)
    mock_kb_instance.add_entity(main_class_entity)
    mock_kb_instance.add_entity(main_func_entity)
    main_py_file_entity.children_ids = [main_class_entity.id, main_func_entity.id]


    # File 2: utils.py (empty for this test, or with minimal structure)
    utils_py_path = "project/src/utils.py"
    utils_py_file_entity = CodeEntity(id=f"{utils_py_path}::{utils_py_path}", entity_type="file", name="utils.py", qualified_name=utils_py_path, language="python", filepath=utils_py_path, start_line=1, end_line=1, raw_text_snippet="# Utility functions here")
    mock_kb_instance.add_entity(utils_py_file_entity)


    # Test case 1: Explain main.py
    initial_state_main: RepoExplainerState = {
        "user_query": "Explain src/main.py", # Not directly used by this node
        "current_focus_path": main_py_path,
        "abstract_code_kb": mock_kb_instance,
    }
    print(f"\n--- Testing file_explainer_node for: {main_py_path} ---")
    result_state_main = file_explainer_node(initial_state_main)
    print(f"Generated Explanation (main.py):\n{result_state_main.get('generated_explanation')}")
    assert result_state_main.get("generated_explanation") is not None
    assert "simulated file explanation" in result_state_main.get("generated_explanation", "").lower()
    assert result_state_main.get("error_message") is None

    # Test case 2: Explain utils.py (which has minimal structure in mock)
    initial_state_utils: RepoExplainerState = {
        "user_query": "Explain utils.py",
        "current_focus_path": utils_py_path,
        "abstract_code_kb": mock_kb_instance,
    }
    print(f"\n--- Testing file_explainer_node for: {utils_py_path} ---")
    result_state_utils = file_explainer_node(initial_state_utils)
    print(f"Generated Explanation (utils.py):\n{result_state_utils.get('generated_explanation')}")
    assert result_state_utils.get("generated_explanation") is not None
    # The placeholder LLM might still give a generic "simulated file explanation"
    # A real LLM would comment on its emptiness or simple content.

    # Test case 3: Target file specified by target_entity_ids (file entity)
    initial_state_target_id: RepoExplainerState = {
        "user_query": "Explain this file.",
        "target_entity_ids": [main_py_file_entity.id], # ID of the file entity for main.py
        "abstract_code_kb": mock_kb_instance,
    }
    print(f"\n--- Testing file_explainer_node via target_entity_id for: {main_py_path} ---")
    result_state_target_id = file_explainer_node(initial_state_target_id)
    print(f"Generated Explanation (target_id main.py):\n{result_state_target_id.get('generated_explanation')}")
    assert "simulated file explanation" in result_state_target_id.get("generated_explanation", "").lower()


    # Test case 4: No file specified
    initial_state_no_file: RepoExplainerState = {
        "user_query": "Explain.",
        "abstract_code_kb": mock_kb_instance,
        # current_focus_path and target_entity_ids are missing or don't point to a file
    }
    print(f"\n--- Testing file_explainer_node (no file specified) ---")
    result_state_no_file = file_explainer_node(initial_state_no_file)
    print(f"Generated Explanation (no file):\n{result_state_no_file.get('generated_explanation')}")
    assert "Error: No file was specified" in result_state_no_file.get("generated_explanation", "")
    assert result_state_no_file.get("error_message") is not None
    
    # Test case 5: File not in KB
    initial_state_file_not_in_kb: RepoExplainerState = {
        "user_query": "Explain non_existent.py",
        "current_focus_path": "project/non_existent.py",
        "abstract_code_kb": mock_kb_instance,
    }
    print(f"\n--- Testing file_explainer_node (file not in KB) ---")
    result_state_file_not_in_kb = file_explainer_node(initial_state_file_not_in_kb)
    print(f"Generated Explanation (file not in KB):\n{result_state_file_not_in_kb.get('generated_explanation')}")
    assert "Could not find detailed information" in result_state_file_not_in_kb.get("generated_explanation", "")

    print("\nfile_explainer_node tests completed.")
