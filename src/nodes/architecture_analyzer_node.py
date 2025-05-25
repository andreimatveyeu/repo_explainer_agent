from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState
from src.core.models import AbstractCodeKnowledgeBase
from src.utils.llm_utils import call_llm_for_summary

def architecture_analyzer_node(state: RepoExplainerState) -> RepoExplainerState:
    """
    Analyzes the repository's structure and components to generate
    a high-level architectural overview.
    Updates `generated_explanation` in the state.
    """
    print("--- Running Node: architecture_analyzer_node ---")
    
    code_kb: Optional[AbstractCodeKnowledgeBase] = state.get("abstract_code_kb")
    overall_summary: Optional[str] = state.get("overall_summary")
    repo_url: Optional[str] = state.get("repo_url")
    local_repo_path: Optional[str] = state.get("local_repo_path")

    updated_state: Dict[str, Any] = {}
    
    if not code_kb:
        message = "Code knowledge base not available. Cannot perform architectural analysis."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        # Keep existing generated_explanation or overall_summary if any
        updated_state["generated_explanation"] = state.get("generated_explanation") or state.get("overall_summary") or "Could not perform architectural analysis due to missing code knowledge base."
        return {**state, **updated_state}

    context_chunks: List[str] = []
    repo_identifier = repo_url or local_repo_path or "the repository"

    context_chunks.append(f"Analyzing the architecture of {repo_identifier}.")
    if overall_summary:
        context_chunks.append(f"Previously generated overall summary:\n{overall_summary}")

    context_chunks.append("\nKey information from the codebase for architectural analysis:")
    
    # File structure overview
    if code_kb.file_paths:
        context_chunks.append("\nFile Structure (Top-level and key directories):")
        # Simple heuristic: show top-level files and directories
        # A more sophisticated approach would identify key architectural layers or components from paths
        top_level_items = set()
        for path in code_kb.file_paths:
            parts = path.split('/')
            if len(parts) == 1:
                top_level_items.add(parts[0])
            else:
                top_level_items.add(parts[0] + "/") # Indicate directory
        
        # Limit the number of items shown to keep context manageable
        max_items_to_show = 15
        if len(top_level_items) > max_items_to_show:
             context_chunks.append("\n".join(sorted(list(top_level_items))[:max_items_to_show]))
             context_chunks.append("... and more.")
        else:
            context_chunks.append("\n".join(sorted(list(top_level_items))))


    # Key code entities (classes, functions)
    key_entities_summary: List[str] = []
    entity_count = 0
    max_entities_to_list = 10 # Limit for brevity in prompt
    
    # Prioritize entities from more "central" files if possible (e.g. not tests, not __init__)
    # This is a simple heuristic; a better one would involve centrality metrics or main module detection
    sorted_entities = sorted(code_kb.entities.values(), key=lambda e: (
        0 if 'test' not in e.filepath.lower() and '__init__' not in e.filepath.lower() else 1,
        e.filepath, 
        e.start_line
    ))

    for entity in sorted_entities:
        if entity.entity_type in ["class", "function"] and entity.name: # Avoid unnamed or file entities here
            # Basic check to avoid overly generic names if possible, or very short names
            if len(entity.name) > 2 and not entity.name.startswith("_"):
                 key_entities_summary.append(f"- {entity.entity_type.capitalize()} '{entity.name}' in '{entity.filepath}'")
                 entity_count += 1
        if entity_count >= max_entities_to_list:
            break
            
    if key_entities_summary:
        context_chunks.append("\nKey Code Entities (selection):")
        context_chunks.extend(key_entities_summary)
        if entity_count >= max_entities_to_list and len(code_kb.entities) > max_entities_to_list:
            context_chunks.append("... and other entities.")

    # Documentation summary (if available and relevant)
    # For now, we assume overall_summary might include this, or it's part of code_kb.docs
    # This could be expanded by summarizing READMEs or other high-level docs.

    prompt = (
        "Based on the provided context (overall summary, file structure, key code entities), "
        "generate a concise architectural overview of the software repository. "
        "Focus on:\n"
        "1. The main purpose and high-level functionality.\n"
        "2. Key components or modules and their primary responsibilities.\n"
        "3. How these components might interact (if inferable).\n"
        "4. Any apparent architectural patterns or design choices (e.g., CLI tool, library, web service).\n"
        "Avoid going into line-by-line code details; provide a structural, high-level understanding."
    )

    architectural_explanation = call_llm_for_summary(
        prompt,
        context_chunks=context_chunks,
        max_tokens=800, # Allow for a more detailed explanation
        temperature=0.3 # Slightly more creative for synthesis
    )

    if architectural_explanation and architectural_explanation.strip():
        print(f"Generated architectural explanation: {architectural_explanation[:200]}...")
        updated_state["generated_explanation"] = architectural_explanation
    else:
        message = "LLM failed to generate an architectural explanation."
        print(f"Warning: {message}")
        # Fallback to overall_summary if architectural analysis fails
        updated_state["generated_explanation"] = overall_summary or "Could not generate an architectural explanation."
        if not state.get("error_message"): # Don't overwrite a more specific error
            updated_state["error_message"] = message 

    return {**state, **updated_state}

if __name__ == '__main__':
    # Example Usage (requires mock objects or a more elaborate setup)
    from src.core.models import CodeEntity

    mock_kb = AbstractCodeKnowledgeBase()
    mock_kb.file_paths = [
        "src/main.py", "src/core/processor.py", "src/utils/helpers.py", 
        "tests/test_main.py", "README.md", "LICENSE", "docs/architecture.md"
    ]
    mock_kb.add_entity(CodeEntity(id="e1", entity_type="file", name="main.py", filepath="src/main.py", language="python", start_line=0, end_line=0))
    mock_kb.add_entity(CodeEntity(id="e2", entity_type="class", name="MainApp", filepath="src/main.py", language="python", start_line=5, end_line=20))
    mock_kb.add_entity(CodeEntity(id="e3", entity_type="function", name="run_processing", filepath="src/core/processor.py", language="python", start_line=10, end_line=30))
    mock_kb.add_entity(CodeEntity(id="e4", entity_type="class", name="DataProcessor", filepath="src/core/processor.py", language="python", start_line=35, end_line=50))
    mock_kb.add_entity(CodeEntity(id="e5", entity_type="function", name="format_data", filepath="src/utils/helpers.py", language="python", start_line=3, end_line=15))
    
    # Add more entities to test selection logic
    for i in range(15):
        mock_kb.add_entity(CodeEntity(id=f"extra_e{i}", entity_type="function", name=f"utility_func_{i}", filepath=f"src/utils/more_utils_{i//5}.py", language="python", start_line=1, end_line=5))


    initial_state: RepoExplainerState = {
        "repo_url": "https_example_com_test_repo_git",
        "abstract_code_kb": mock_kb,
        "overall_summary": "This is a Python project for data processing with a CLI interface.",
        "user_query": "Explain the architecture of this repository.",
        "explanation_history": []
    }

    print("\n--- Testing architecture_analyzer_node ---")
    result_state = architecture_analyzer_node(initial_state)
    
    print(f"\nGenerated Explanation:\n{result_state.get('generated_explanation')}")
    if result_state.get("error_message"):
        print(f"Error: {result_state.get('error_message')}")

    # A basic assertion (actual LLM output will vary)
    assert result_state.get("generated_explanation") is not None
    assert len(result_state.get("generated_explanation", "")) > len(initial_state.get("overall_summary", "")) \
        or "architectural analysis" in result_state.get("generated_explanation", "").lower() \
        or "components" in result_state.get("generated_explanation", "").lower()

    print("\narchitecture_analyzer_node test completed.")
