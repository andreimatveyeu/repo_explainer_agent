import os
import subprocess
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
            # Define a base directory for clones, e.g., './cloned_repositories'
            # This should be configurable.
            clone_base_dir = os.path.abspath(os.path.join(os.getcwd(), "cloned_repositories"))
            os.makedirs(clone_base_dir, exist_ok=True)
            
            repo_name = get_repo_name_from_url(repo_url)
            target_clone_path = os.path.join(clone_base_dir, repo_name)

            # If a repo_url is provided, we prioritize cloning it.
            # A more robust solution might involve removing the old directory if it exists and isn't the correct repo,
            # or implementing a git pull strategy if it is the correct repo.
            # For now, if the directory exists, we'll assume it's potentially the target repo and could be used or updated.
            # If it doesn't exist, we clone.
            if os.path.isdir(target_clone_path):
                # Consider adding logic to verify if this directory is indeed the repo from repo_url
                # or if it needs to be updated (e.g., git pull).
                # For this fix, we'll assume if it exists, it might be stale but we'll proceed.
                # A cleaner approach for "always clone if URL is given" might be to remove target_clone_path first.
                # However, to minimize changes, we'll keep the "use if exists" but ensure cloning happens if not.
                print(f"Directory {target_clone_path} exists. Checking if it's the intended repo or needs cloning.")
                # For simplicity in this fix, if we want to ensure the URL is cloned,
                # we should ideally clone to a fresh path or ensure the existing one is correct/updated.
                # Let's refine to: if it exists, print a message but still treat it as the path.
                # If it doesn't exist, clone. This matches original behavior if path existed, but now repo_url is primary.
                # A more aggressive strategy would be to remove and re-clone, or `git pull`.
                # The original code for existing path:
                # print(f"Repository already cloned at: {target_clone_path}. Using existing.")
                # Let's stick to a similar behavior for now if it exists.
                if not os.path.exists(os.path.join(target_clone_path, ".git")): # Basic check if it's a git repo
                    print(f"Directory {target_clone_path} exists but is not a git repository or is incomplete. Attempting to clone...")
                    # Potentially remove target_clone_path here if it's not a valid git repo to avoid clone errors.
                    # import shutil
                    # shutil.rmtree(target_clone_path) # Be cautious with this
                    # os.makedirs(target_clone_path) # Recreate after delete if git clone expects parent dir
                    result = subprocess.run(
                        ["git", "clone", repo_url, target_clone_path], 
                        check=True, capture_output=True, text=True
                    )
                    print(f"Clone successful. Output:\n{result.stdout}")
                else:
                    print(f"Repository already cloned at: {target_clone_path}. Using existing. Consider 'git pull' for updates.")
                    # Optionally, could add logic here to pull latest changes:
                    # print("Attempting to pull latest changes...")
                    # subprocess.run(["git", "pull"], cwd=target_clone_path, check=True, capture_output=True, text=True)

            else:
                print(f"Cloning repository from {repo_url} to {target_clone_path}...")
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
    
    # Cleanup cloned_repositories directory if you want a clean slate for each test run
    # cloned_base = os.path.abspath(os.path.join(os.getcwd(), "cloned_repositories"))
    # if os.path.exists(cloned_base):
    #     print(f"Cleaning up {cloned_base}...")
    #     shutil.rmtree(cloned_base)

    print("\ninitialize_repository node tests completed.")
