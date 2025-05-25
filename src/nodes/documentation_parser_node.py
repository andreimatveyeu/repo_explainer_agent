import os
from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState

# Common documentation file names and directories
COMMON_DOC_FILES = [
    "README.md", "README.rst", "README.txt", "README",
    "INSTALL.md", "INSTALL.rst", "INSTALL.txt",
    "CONTRIBUTING.md", "CONTRIBUTING.rst",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    "CHANGELOG.md", "CHANGELOG.rst",
    "TODO.md", "TODO.txt"
]
COMMON_DOC_DIRS = ["docs", "doc", "documentation"]
COMMON_DOC_EXTENSIONS = [".md", ".rst", ".txt", ".adoc", ".asciidoc"] # Common text-based doc formats

def documentation_parser(state: RepoExplainerState) -> RepoExplainerState:
    """
    Parses common documentation files from the repository.
    Scans for READMEs, files in 'docs/' directories, and other common documentation files.
    Updates `parsed_documentation` in the state with {file_path: content}.
    """
    print("--- Running Node: documentation_parser ---")
    local_repo_path: Optional[str] = state.get("local_repo_path")
    parsed_docs: Dict[str, str] = state.get("parsed_documentation", {}) # Initialize or get existing
    updated_state: Dict[str, Any] = {}

    if not local_repo_path or not os.path.isdir(local_repo_path):
        message = f"Invalid or missing local_repo_path: {local_repo_path}. Cannot parse documentation."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        # Ensure parsed_documentation is at least an empty dict if not already
        updated_state["parsed_documentation"] = parsed_docs 
        return {**state, **updated_state}

    print(f"Scanning for documentation files in: {local_repo_path}")
    
    files_to_parse: List[str] = []

    # 1. Check for common top-level documentation files
    for doc_file_name in COMMON_DOC_FILES:
        potential_file_path = os.path.join(local_repo_path, doc_file_name)
        if os.path.isfile(potential_file_path):
            files_to_parse.append(potential_file_path)
            print(f"Found top-level doc file: {potential_file_path}")

    # 2. Scan common documentation directories
    for doc_dir_name in COMMON_DOC_DIRS:
        potential_doc_dir = os.path.join(local_repo_path, doc_dir_name)
        if os.path.isdir(potential_doc_dir):
            print(f"Scanning documentation directory: {potential_doc_dir}")
            for root, _, files in os.walk(potential_doc_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    _, extension = os.path.splitext(file_name)
                    if extension.lower() in COMMON_DOC_EXTENSIONS:
                        files_to_parse.append(file_path)
                        print(f"Found doc file in dir: {file_path}")
    
    # Ensure uniqueness
    files_to_parse = sorted(list(set(files_to_parse)))

    for file_path in files_to_parse:
        if file_path in parsed_docs: # Avoid re-parsing if already present
            print(f"Skipping already parsed doc file: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            parsed_docs[os.path.abspath(file_path)] = content # Store with absolute path
            print(f"Successfully parsed: {file_path} ({len(content)} chars)")
        except Exception as e:
            print(f"Error reading documentation file {file_path}: {e}")
            # Optionally, store error info in parsed_docs or a separate error log
            parsed_docs[os.path.abspath(file_path)] = f"Error reading file: {str(e)}"

    updated_state["parsed_documentation"] = parsed_docs
    if "error_message" not in updated_state: # Don't overwrite a previous critical error
         updated_state["error_message"] = None
    
    print(f"Documentation parsing completed. Found {len(parsed_docs)} documents.")
    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage (for testing the node directly)
    test_repo_dir = "./temp_doc_parser_test_repo"
    os.makedirs(os.path.join(test_repo_dir, "docs", "subdir"), exist_ok=True)
    os.makedirs(os.path.join(test_repo_dir, "another_docs"), exist_ok=True) # Test non-standard doc dir

    with open(os.path.join(test_repo_dir, "README.md"), "w") as f:
        f.write("# Main Readme\nContent of main readme.")
    with open(os.path.join(test_repo_dir, "LICENSE"), "w") as f:
        f.write("MIT License Content.")
    with open(os.path.join(test_repo_dir, "docs", "guide.rst"), "w") as f:
        f.write("Guide in RST format.\n================")
    with open(os.path.join(test_repo_dir, "docs", "subdir", "api.txt"), "w") as f:
        f.write("API details in text file.")
    with open(os.path.join(test_repo_dir, "src", "code.py"), "w") as f: # Non-doc file
        f.write("print('hello')")


    initial_state_no_docs: RepoExplainerState = {
        "user_query": "Parse docs.",
        "local_repo_path": test_repo_dir
        # "parsed_documentation" is not present initially
    }
    print("\nTesting documentation_parser (initial run):")
    result_state = documentation_parser(initial_state_no_docs)
    
    assert result_state.get("error_message") is None
    parsed_content = result_state.get("parsed_documentation", {})
    print(f"Parsed documents ({len(parsed_content)}):")
    for path, content_preview in parsed_content.items():
        print(f"  {path}: {content_preview[:50].replace(os.linesep, ' ')}...")

    expected_doc_count = 4 # README.md, LICENSE, docs/guide.rst, docs/subdir/api.txt
    assert len(parsed_content) == expected_doc_count
    assert os.path.abspath(os.path.join(test_repo_dir, "README.md")) in parsed_content
    assert "Main Readme" in parsed_content[os.path.abspath(os.path.join(test_repo_dir, "README.md"))]

    # Test with pre-existing parsed_documentation (should not re-parse)
    initial_state_with_docs: RepoExplainerState = {
        "user_query": "Parse docs again.",
        "local_repo_path": test_repo_dir,
        "parsed_documentation": {
            os.path.abspath(os.path.join(test_repo_dir, "README.md")): "Already parsed content."
        }
    }
    print("\nTesting documentation_parser (with pre-existing docs):")
    result_state_again = documentation_parser(initial_state_with_docs)
    parsed_content_again = result_state_again.get("parsed_documentation", {})
    
    assert len(parsed_content_again) == expected_doc_count # Still same total, but one was skipped
    assert parsed_content_again[os.path.abspath(os.path.join(test_repo_dir, "README.md"))] == "Already parsed content."
    assert "Guide in RST" in parsed_content_again[os.path.abspath(os.path.join(test_repo_dir, "docs", "guide.rst"))]


    # Test with invalid path
    initial_state_invalid_path: RepoExplainerState = {
        "user_query": "Parse docs.",
        "local_repo_path": "./non_existent_path_for_docs"
    }
    print("\nTesting documentation_parser (invalid path):")
    result_state_invalid = documentation_parser(initial_state_invalid_path)
    assert result_state_invalid.get("error_message") is not None
    assert len(result_state_invalid.get("parsed_documentation", {})) == 0


    # Cleanup
    if os.path.exists(test_repo_dir):
        import shutil
        shutil.rmtree(test_repo_dir)

    print("\ndocumentation_parser node tests completed.")
