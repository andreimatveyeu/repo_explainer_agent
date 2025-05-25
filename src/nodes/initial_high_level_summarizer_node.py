import os
from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState
from src.core.models import AbstractCodeKnowledgeBase
from src.utils.llm_utils import call_llm_for_summary

MAX_CONTEXT_CHUNKS_FOR_SUMMARY = 10 # Limit how much raw doc/code content we feed in one go
MAX_CHUNK_LENGTH = 2000 # Max characters per chunk of context

def format_kb_overview_for_llm(code_kb: AbstractCodeKnowledgeBase) -> List[str]:
    """
    Formats a brief overview of the AbstractCodeKnowledgeBase for an LLM prompt.
    """
    if not code_kb:
        return ["No code structure information available."]

    overview = []
    overview.append(f"Repository contains {len(code_kb.file_paths)} files.")
    if code_kb.detected_languages:
        overview.append(f"Detected languages: {', '.join(code_kb.detected_languages)}.")

    # Highlight key files or entities (e.g., top-level files, files with many children)
    # This is a simple heuristic; more sophisticated selection could be used.
    top_level_files = [fp for fp in code_kb.file_paths if os.path.dirname(fp) == code_kb.entities.get(f"{fp}::{fp}").filepath] # Assuming file entity ID is filepath::filepath
    
    # Get a few key entities (e.g., classes and functions in top-level modules)
    key_entities_info = []
    entities_to_show = 0
    max_entities_to_list = 5 # Limit how many entities we list to keep prompt concise

    for entity_id, entity in code_kb.entities.items():
        if entity.entity_type in ["class", "function"] and entity.filepath in top_level_files:
            if entities_to_show < max_entities_to_list:
                key_entities_info.append(f"- {entity.entity_type} '{entity.name}' in '{entity.filepath}'")
                entities_to_show +=1
            else:
                break
    
    if key_entities_info:
        overview.append("Some key code structures identified:")
        overview.extend(key_entities_info)
    elif code_kb.entities:
        overview.append(f"Total code entities found: {len(code_kb.entities)}.")
        
    return ["\n".join(overview)]


def format_documentation_for_llm(parsed_docs: Dict[str, str]) -> List[str]:
    """
    Formats parsed documentation for an LLM prompt, taking snippets.
    """
    if not parsed_docs:
        return ["No documentation files parsed."]
    
    doc_snippets = []
    doc_snippets.append("Key Documentation Files:")
    for i, (path, content) in enumerate(parsed_docs.items()):
        if i >= MAX_CONTEXT_CHUNKS_FOR_SUMMARY / 2: # Limit doc files shown
            doc_snippets.append("... and other documentation files.")
            break
        snippet = content[:MAX_CHUNK_LENGTH // 2] # Take a smaller snippet from each doc
        if len(content) > MAX_CHUNK_LENGTH // 2:
            snippet += "..."
        doc_snippets.append(f"\n--- Content from: {os.path.basename(path)} ---\n{snippet}")
    return ["\n".join(doc_snippets)]


def initial_high_level_summarizer(state: RepoExplainerState) -> RepoExplainerState:
    """
    Generates a high-level summary of the repository using an LLM.
    It uses information from `abstract_code_kb` and `parsed_documentation`.
    Updates `overall_summary` in the state.
    """
    print("--- Running Node: initial_high_level_summarizer ---")
    code_kb: Optional[AbstractCodeKnowledgeBase] = state.get("abstract_code_kb")
    parsed_docs: Optional[Dict[str, str]] = state.get("parsed_documentation")
    updated_state: Dict[str, Any] = {}

    if not code_kb and not parsed_docs:
        message = "Cannot generate summary: No code knowledge base or parsed documentation available."
        print(f"Warning: {message}")
        updated_state["overall_summary"] = "No information available to generate a repository summary."
        # updated_state["error_message"] = message # This might be too strong for just missing data
        return {**state, **updated_state}

    context_chunks: List[str] = []
    if code_kb:
        context_chunks.extend(format_kb_overview_for_llm(code_kb))
    if parsed_docs:
        context_chunks.extend(format_documentation_for_llm(parsed_docs))
    
    prompt = (
        "You are an expert software engineering assistant. Based on the following information "
        "about a software repository (code structure overview and snippets from documentation files), "
        "provide a concise, high-level summary. The summary should cover:\n"
        "1. The likely overall purpose or domain of the repository.\n"
        "2. The main programming languages or technologies detected.\n"
        "3. Key directory structures or important modules, if apparent.\n"
        "4. Any notable features or functionalities mentioned in the documentation.\n\n"
        "Generate a summary that would be helpful for someone trying to quickly understand what this repository is about."
    )

    summary = call_llm_for_summary(prompt, context_chunks=context_chunks, max_tokens=1000)

    if summary:
        print(f"Generated overall summary: {summary[:200]}...")
        updated_state["overall_summary"] = summary
        updated_state["error_message"] = None
    else:
        message = "Failed to generate overall summary using LLM."
        print(f"Error: {message}")
        updated_state["overall_summary"] = "Could not generate a summary for the repository at this time."
        updated_state["error_message"] = message
        
    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage
    import os # Required for os.path.basename in format_documentation_for_llm
    
    # Mock CodeKnowledgeBase
    mock_kb = AbstractCodeKnowledgeBase()
    mock_kb.file_paths = ["project/src/main.py", "project/src/utils.py", "project/README.md"]
    mock_kb.detected_languages = ["python"]
    # Add a dummy file entity for format_kb_overview_for_llm to work
    mock_kb.add_entity(CodeEntity(id="project/src/main.py::project/src/main.py", entity_type="file", name="main.py", qualified_name="project/src/main.py", language="python", filepath="project/src/main.py", start_line=1, end_line=10))

    mock_kb.add_entity(CodeEntity(id="project/src/main.py::MyClass", entity_type="class", name="MyClass", qualified_name="main.MyClass", language="python", filepath="project/src/main.py", start_line=1, end_line=10))
    mock_kb.add_entity(CodeEntity(id="project/src/utils.py::helper_func", entity_type="function", name="helper_func", qualified_name="utils.helper_func", language="python", filepath="project/src/utils.py", start_line=1, end_line=5))

    # Mock Parsed Documentation
    mock_docs = {
        "project/README.md": "# Project Alpha\nThis project does amazing things with data.",
        "project/docs/USAGE.md": "## Usage\nTo use this project, first install dependencies..."
    }

    initial_state_full: RepoExplainerState = {
        "user_query": "Summarize this repo.", # Not directly used by this node, but part of state
        "abstract_code_kb": mock_kb,
        "parsed_documentation": mock_docs
    }
    print("\nTesting initial_high_level_summarizer (full data):")
    result_state_full = initial_high_level_summarizer(initial_state_full)
    print(f"Resulting summary: {result_state_full.get('overall_summary')}")
    assert result_state_full.get("overall_summary") is not None
    assert "simulated high-level summary" in result_state_full.get("overall_summary", "").lower()


    initial_state_no_kb: RepoExplainerState = {
        "user_query": "Summarize.",
        "parsed_documentation": mock_docs
        # "abstract_code_kb" is missing
    }
    print("\nTesting initial_high_level_summarizer (no KB):")
    result_state_no_kb = initial_high_level_summarizer(initial_state_no_kb)
    print(f"Resulting summary (no KB): {result_state_no_kb.get('overall_summary')}")
    assert result_state_no_kb.get("overall_summary") is not None
    
    initial_state_no_docs: RepoExplainerState = {
        "user_query": "Summarize.",
        "abstract_code_kb": mock_kb
        # "parsed_documentation" is missing
    }
    print("\nTesting initial_high_level_summarizer (no docs):")
    result_state_no_docs = initial_high_level_summarizer(initial_state_no_docs)
    print(f"Resulting summary (no docs): {result_state_no_docs.get('overall_summary')}")
    assert result_state_no_docs.get("overall_summary") is not None

    initial_state_empty: RepoExplainerState = {
        "user_query": "Summarize.",
        # Both KB and docs missing
    }
    print("\nTesting initial_high_level_summarizer (empty state):")
    result_state_empty = initial_high_level_summarizer(initial_state_empty)
    print(f"Resulting summary (empty): {result_state_empty.get('overall_summary')}")
    assert "No information available" in result_state_empty.get("overall_summary", "")

    print("\ninitial_high_level_summarizer node tests completed.")
