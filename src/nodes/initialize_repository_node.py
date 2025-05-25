import os
import subprocess
import tempfile
import shutil
import atexit
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from src.core.state import RepoExplainerState

def get_repo_name_from_url(url: str) -> str:
    """Extracts a repository name from a Git URL."""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) > 0:
        repo_name_with_ext = path_parts[-1]
        return repo_name_with_ext.replace('.git', '')
    return "cloned_repo" # Fallback name

def initialize_repository(state: RepoExplainerState) -> RepoExplainerState:
    """
    Initializes the repository.
    If `repo_url` is provided, it attempts to clone the repository to a local directory.
    If `local_repo_path` is provided and valid, it uses that.
    Updates `local_repo_path` and potentially `error_message` in the state.
    """
    print("--- Running Node: initialize_repository ---")
    repo_url: Optional[str] = state.get("repo_url")
    local_repo_path: Optional[str] = state.get("local_repo_path")
    updated_state: Dict[str, Any] = {}

    if repo_url:
        # Basic git clone functionality.
        # For a production system, use a library like GitPython or handle errors more robustly.
        try:
            # Create a unique temporary directory for this clone session.
            # The actual repo will be cloned as a subdirectory within this temp_dir.
            # tempfile.mkdtemp creates a directory with a unique name.
            temp_parent_dir = tempfile.mkdtemp(prefix="repo_explainer_clone_")
            
            # Register a cleanup function for the parent temporary directory.
            # This ensures the entire directory created by mkdtemp (and its contents) 
            # is removed when the Python interpreter exits.
            # ignore_errors=True helps prevent cleanup issues if, for example, a file is locked.
            atexit.register(shutil.rmtree, temp_parent_dir, ignore_errors=True)
            print(f"Created temporary directory for clone: {temp_parent_dir}. It will be cleaned up on exit.")

            repo_name = get_repo_name_from_url(repo_url)
            # The repository will be cloned into a subdirectory named `repo_name` inside `temp_parent_dir`.
            target_clone_path = os.path.join(temp_parent_dir, repo_name)

            print(f"Cloning repository from {repo_url} into temporary directory at {target_clone_path}...")
            # The `git clone` command will create the `repo_name` directory within `temp_parent_dir`.
            result = subprocess.run(
                ["git", "clone", repo_url, target_clone_path], 
                check=True, capture_output=True, text=True
            )
            print(f"Clone successful. Output:\n{result.stdout}")
            updated_state["local_repo_path"] = target_clone_path

        except subprocess.CalledProcessError as e:
            message = f"Failed to clone repository from {repo_url}. Error: {e.stderr}"
            print(f"Error: {message}")
            updated_state["error_message"] = message
            return {**state, **updated_state} # Return early on error
        except Exception as e:
            message = f"An unexpected error occurred during repository cloning: {str(e)}"
            print(f"Error: {message}")
            updated_state["error_message"] = message
            return {**state, **updated_state} # Return early on error
    elif local_repo_path:
        # This block is now executed only if repo_url was NOT provided.
        if os.path.isdir(local_repo_path):
            print(f"Using existing local repository path (repo_url not provided): {local_repo_path}")
            updated_state["local_repo_path"] = os.path.abspath(local_repo_path)
            # We might want to ensure it's a git repo or pull latest here in a more advanced version
        else:
            message = f"Provided local_repo_path '{local_repo_path}' is not a valid directory (repo_url not provided)."
            print(f"Error: {message}")
            updated_state["error_message"] = message
            return {**state, **updated_state} # Return early on error
    else:
        # Neither repo_url nor local_repo_path provided.
        message = "Neither repo_url nor local_repo_path provided in the state."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        return {**state, **updated_state}

    # Clear any previous error message if successful
    if "error_message" not in updated_state:
        updated_state["error_message"] = None 

    return {**state, **updated_state}

if __name__ == '__main__':
    # Example Usage (for testing the node directly)
    
    # Test case 1: Valid local path
    # Create a dummy directory for testing
    if not os.path.exists("./temp_test_repo"):
        os.makedirs("./temp_test_repo/.git") # Make it look like a git repo
    
    initial_state_local: RepoExplainerState = {
        "user_query": "Explain this local repo.",
        "local_repo_path": "./temp_test_repo"
    }
    print(f"\nTesting with local path: {initial_state_local.get('local_repo_path')}")
    result_state_local = initialize_repository(initial_state_local)
    print(f"Resulting state (local): {result_state_local}")
    assert result_state_local.get("local_repo_path") is not None
    assert result_state_local.get("error_message") is None

    # Test case 2: Invalid local path
    initial_state_invalid_local: RepoExplainerState = {
        "user_query": "Explain this non-existent repo.",
        "local_repo_path": "./non_existent_repo_path"
    }
    print(f"\nTesting with invalid local path: {initial_state_invalid_local.get('local_repo_path')}")
    result_state_invalid_local = initialize_repository(initial_state_invalid_local)
    print(f"Resulting state (invalid local): {result_state_invalid_local}")
    assert result_state_invalid_local.get("error_message") is not None
    
    # Test case 3: Git URL (requires git to be installed and a valid public repo URL)
    # Using a small, public repository for testing.
    # Replace with a more stable test repo if needed.
    test_repo_url = "https://github.com/git-fixtures/basic.git" 
    # test_repo_url = "https://github.com/langchain-ai/langgraph.git" # A larger repo
    
    initial_state_url: RepoExplainerState = {
        "user_query": "Explain this repo from URL.",
        "repo_url": test_repo_url
    }
    print(f"\nTesting with Git URL: {test_repo_url}")
    result_state_url = initialize_repository(initial_state_url)
    print(f"Resulting state (URL): {result_state_url}")
    if result_state_url.get("local_repo_path"):
        assert os.path.isdir(result_state_url["local_repo_path"])
        print(f"Cloned to: {result_state_url['local_repo_path']}")
    else:
        print(f"Cloning failed or skipped, error: {result_state_url.get('error_message')}")

    # Cleanup dummy directory
    if os.path.exists("./temp_test_repo"):
        import shutil
        shutil.rmtree("./temp_test_repo")
    
    # Temporary directories used for cloning (if any test_repo_url was used) 
    # will be cleaned up automatically via the atexit handler registered in the function.
    # No manual cleanup of a "cloned_repositories" directory is needed here anymore.

    print("\ninitialize_repository node tests completed.")
