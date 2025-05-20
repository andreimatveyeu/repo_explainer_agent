import os
import json
import shutil
import re
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Literal
import requests # For calling local Ollama
import google.generativeai as genai # For calling Google Gemini
from langgraph.graph import StateGraph, END
#from langgraph.checkpoint.sqlite import SqliteSaver # For persistence if needed

# Import tools from the sibling tools.py file
from . import tools

# --- Configuration ---
TEMP_REPO_DIR = "temp_cloned_repo"
MAX_FILES_TO_PROCESS_PER_BATCH = 10 # For iterative processing
MAX_ITERATIONS_METADATA_EXTRACTION = 5 # Safety break for loops
LLM_SUMMARY_CHAR_LIMIT = 2000 # Max chars to send to mock LLM for one item
FILE_PREVIEW_CHAR_LIMIT = 1000 # For reading generic files if not specifically parsed

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Defines the state of the repository explainer agent.

    This TypedDict holds all the information that is passed between the nodes
    of the LangGraph. It includes inputs, intermediate processing results,
    extracted data, summaries, and control flow variables.
    """
    # Inputs
    repo_url_or_path: str
    is_local_path: bool

    # Paths and File Lists
    local_repo_path: Optional[str]
    directory_tree: Optional[List[str]]
    all_files_in_repo: List[str] # Flat list of all file paths relative to repo root

    readme_files: List[str]
    config_files: List[str] # General config files
    dependency_files: List[str]
    source_code_files: List[str] # Files like .py, .java, .js etc.
    
    # Processing Queues and Tracking
    files_to_parse_queue: List[str] # Current batch of files to parse for metadata/content
    processed_files_this_iteration: List[str]
    processed_metadata_files: List[str] # Files whose metadata has been extracted
    
    # Extracted Data
    parsed_readme_data: List[Dict[str, Any]] # Data from parse_markdown_file
    parsed_config_data: List[Dict[str, Any]] # Data from parse_generic_config_file or specific ones
    extracted_metadata: Dict[str, Dict[str, Any]] # file_path -> parsed_data (e.g., from parse_python_file)
    extracted_dependencies: Dict[str, List[str]] # e.g. {"main": [...], "dev": [...]}
    
    # Summaries (intermediate and final)
    repository_purpose_summary: Optional[str]
    key_components_summary: List[Dict[str, str]] # [{"name": "comp_name", "summary": "...", "source_file": "path"}]
    
    # Control Flow & Iteration
    current_processing_stage: Optional[str] # e.g., "readme", "config", "metadata", "dependencies"
    iteration_count: int
    max_iterations: int
    
    # Final Output
    final_report: Optional[Dict[str, Any]]
    error_messages: List[str]
    status_messages: List[str]


# --- LLM Interaction ---
def call_llm_api(prompt: str, task_description: str, use_local: bool = False) -> str:
    """
    Calls an LLM API to get a response for a given prompt.

    Supports:
    1. Local Ollama server if `use_local` is True and OLLAMA_URL and OLLAMA_MODEL are set.
    2. Google Gemini API if GOOGLE_AISTUDIO_API_KEY is set and `use_local` is False.
    3. Falls back to a placeholder if neither is configured or an error occurs.

    Args:
        prompt: The fully formatted prompt to send to the LLM.
        task_description: A brief description of the task for context (used in placeholder).
        use_local: If True, attempts to use a local Ollama server. Defaults to False.

    Returns:
        A string, which is the LLM's response or a placeholder/error message.
    """
    if use_local:
        ollama_url = os.getenv("OLLAMA_URL")
        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_url and ollama_model:
            try:
                response = requests.post(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=60 # 60 seconds timeout
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except requests.RequestException as e:
                print(f"Error calling local Ollama: {e}")
                return f"[Ollama Call Error for: {task_description} - {e}]"
            except Exception as e:
                print(f"Error processing Ollama response: {e}")
                return f"[Ollama Response Error for: {task_description} - {e}]"
        else:
            print("OLLAMA_URL or OLLAMA_MODEL not set, but use_local=True. Falling back.")

    google_api_key = os.getenv("GOOGLE_AISTUDIO_API_KEY")
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            model_name = "gemini-2.5-flash-preview-04-17"
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Google Gemini API: {e}")
            return f"[Gemini Call Error for: {task_description} - {e}]"

    # Fallback placeholder response
    # print(f"\n--- SIMULATED LLM CALL ({task_description}) ---")
    # print(f"PROMPT:\n{prompt}")
    # print("--- END PROMPT ---")
    return f"[LLM Output Placeholder for: {task_description} - First 100 chars of prompt: {prompt[:100].replace('\n', ' ')}...]"


# --- Helper Functions ---
def _get_file_path(state: AgentState, file_rel_path: str) -> str:
    """
    Constructs the full path to a file within the cloned repository.

    Args:
        state: The current agent state, containing `local_repo_path`.
        file_rel_path: The relative path of the file from the repository root.

    Returns:
        The absolute path to the file.
    """
    return os.path.join(state["local_repo_path"], file_rel_path)

def _add_error(state: AgentState, message: str) -> None:
    """
    Adds an error message to the agent's state and prints it.

    Args:
        state: The current agent state.
        message: The error message to add.
    """
    state["error_messages"].append(message)
    print(f"ERROR: {message}")

def _add_status(state: AgentState, message: str) -> None:
    """
    Adds a status message to the agent's state and prints it.

    Args:
        state: The current agent state.
        message: The status message to add.
    """
    state["status_messages"].append(message)
    print(f"STATUS: {message}")

# --- Graph Node Implementations ---

def initialize_repository_scan(state: AgentState) -> AgentState:
    """
    Initializes the repository scan process.

    This node performs the following actions:
    - Resets relevant parts of the agent state.
    - Clones the remote repository if a URL is provided, or validates the local path.
    - Generates a directory tree of the repository.
    - Identifies and categorizes files (READMEs, configs, dependencies, source code).
    - Sets the initial processing stage.

    Args:
        state: The current agent state, containing `repo_url_or_path` and `is_local_path`.

    Returns:
        The updated agent state.
    """
    # Initialize error_messages and status_messages first
    state["error_messages"] = []
    state["status_messages"] = []
    _add_status(state, "Initializing repository scan...")
    state["parsed_readme_data"] = []
    state["parsed_config_data"] = []
    state["extracted_metadata"] = {}
    state["extracted_dependencies"] = {"main": [], "dev": [], "others": []}
    state["key_components_summary"] = []
    state["repository_purpose_summary"] = "Not yet determined."
    state["iteration_count"] = 0
    state["max_iterations"] = MAX_ITERATIONS_METADATA_EXTRACTION
    state["processed_metadata_files"] = []
    state["all_files_in_repo"] = []


    repo_input = state["repo_url_or_path"]
    
    if not state.get("is_local_path"): # If it's a URL
        _add_status(state, f"Cloning repository from {repo_input} to {TEMP_REPO_DIR}...")
        if os.path.exists(TEMP_REPO_DIR):
            shutil.rmtree(TEMP_REPO_DIR) # Clean up previous run
        os.makedirs(TEMP_REPO_DIR, exist_ok=True)
        success, message_or_path = tools.clone_repo(repo_input, TEMP_REPO_DIR)
        if not success:
            _add_error(state, f"Failed to clone repository: {message_or_path}")
            state["local_repo_path"] = None
            return state
        state["local_repo_path"] = message_or_path
    else: # It's a local path
        if not os.path.isdir(repo_input):
            _add_error(state, f"Provided local path is not a valid directory: {repo_input}")
            state["local_repo_path"] = None
            return state
        state["local_repo_path"] = repo_input
        _add_status(state, f"Using local repository path: {state['local_repo_path']}")

    # Get directory tree
    _add_status(state, "Generating directory tree...")
    state["directory_tree"] = tools.get_directory_tree(state["local_repo_path"])
    # print("\n".join(state["directory_tree"])) # For debugging

    # Identify file types
    state["readme_files"] = []
    state["config_files"] = []
    state["dependency_files"] = []
    state["source_code_files"] = []
    
    # Common file patterns
    readme_patterns = [r"readme(\.(md|rst|txt))?$", r"notice(\.(md|rst|txt))?$", r"license(\.(md|rst|txt))?$"]
    config_patterns = [
        r"docker-compose\.yml$", r"dockerfile$", r"\.env(\.example)?$", r"Makefile$",
        r"Procfile$", r"manifest\.json$", r"^\.?config(\.(json|yaml|yml|toml|ini|xml))?$" # General config.*
    ] 
    # More specific config files that might also be dependency files
    # These are handled by identify_dependencies_from_file but good to list them
    dependency_config_patterns = [
        r"setup\.py$", r"pyproject\.toml$", r"requirements.*\.txt$", r"environment\.yml$",
        r"package\.json$", r"pom\.xml$", r"build\.gradle(\.kts)?$", r"Pipfile$", r"poetry\.lock$", r"package-lock\.json$"
    ]
    source_code_exts = [".py", ".java", ".js", ".ts", ".go", ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".scala", ".kt", ".swift"]

    for root, _, files in os.walk(state["local_repo_path"]):
        if ".git" in root or "__pycache__" in root or "node_modules" in root: # Basic ignore
            continue
        for file in files:
            file_path_abs = os.path.join(root, file)
            file_path_rel = os.path.relpath(file_path_abs, state["local_repo_path"])
            state["all_files_in_repo"].append(file_path_rel)
            
            file_lower = file.lower()
            file_ext = Path(file).suffix.lower()

            matched_readme = any(re.search(p, file_lower, re.IGNORECASE) for p in readme_patterns)
            matched_config = any(re.search(p, file_lower, re.IGNORECASE) for p in config_patterns)
            matched_dep_conf = any(re.search(p, file_lower, re.IGNORECASE) for p in dependency_config_patterns)

            if matched_readme:
                state["readme_files"].append(file_path_rel)
            elif matched_dep_conf: # Prioritize as dependency file
                state["dependency_files"].append(file_path_rel)
                if file_ext in [".toml", ".json", ".xml", ".yml", ".yaml", ".ini"]: # Some dep files are also general configs
                    state["config_files"].append(file_path_rel)
            elif matched_config:
                state["config_files"].append(file_path_rel)
            
            if file_ext in source_code_exts:
                state["source_code_files"].append(file_path_rel)

    # Deduplicate (e.g. pyproject.toml can be config and dependency)
    state["readme_files"] = sorted(list(set(state["readme_files"])))
    state["config_files"] = sorted(list(set(state["config_files"])))
    state["dependency_files"] = sorted(list(set(state["dependency_files"])))
    state["source_code_files"] = sorted(list(set(state["source_code_files"])))
    
    _add_status(state, f"Identified READMEs: {len(state['readme_files'])}")
    _add_status(state, f"Identified Configs: {len(state['config_files'])}")
    _add_status(state, f"Identified Dependency Files: {len(state['dependency_files'])}")
    _add_status(state, f"Identified Source Code Files: {len(state['source_code_files'])}")
    
    state["current_processing_stage"] = "process_readme"
    return state

def process_readme_files(state: AgentState) -> AgentState:
    """
    Processes identified README files in the repository.

    This node iterates through README files, parses them (assuming Markdown),
    and extracts information. It also attempts to derive an initial repository
    purpose summary from the first README's title or introductory content
    using an LLM call.

    Args:
        state: The current agent state, containing `local_repo_path` and `readme_files`.

    Returns:
        The updated agent state with parsed README data and potentially an initial purpose summary.
    """
    if not state["local_repo_path"]: return state # Should not happen if init was successful
    _add_status(state, f"Processing {len(state['readme_files'])} README files...")
    
    for file_rel_path in state["readme_files"]:
        full_path = _get_file_path(state, file_rel_path)
        _add_status(state, f"Parsing README: {file_rel_path}")
        parsed_data = tools.parse_markdown_file(full_path) # Assuming most READMEs are Markdown
        if parsed_data.get("error"):
            _add_error(state, f"Error parsing README {file_rel_path}: {parsed_data['error']}")
        else:
            state["parsed_readme_data"].append(parsed_data)
            # Attempt to set initial repository purpose from the first README's title or first section
            if not state["repository_purpose_summary"] or state["repository_purpose_summary"] == "Not yet determined.":
                if parsed_data.get("title"):
                    title_text = parsed_data["title"]
                    prompt = f"""You are an expert code repository analyst.
Based on the following repository title, provide a detailed (2-3 sentences) summary of the repository's likely purpose.
Consider and include:
- The primary goal or problem it aims to solve.
- Potential main technologies or programming languages implied.
- The intended audience or type of user, if inferable.

Repository Title:
---
{title_text}
---

Detailed Purpose Summary:"""
                    state["repository_purpose_summary"] = call_llm_api(prompt, f"Summarize repository purpose from title '{title_text[:50]}...'")
                elif parsed_data.get("sections") and parsed_data["sections"][0].get("content_preview"):
                    intro_text = parsed_data["sections"][0]["content_preview"]
                    prompt = f"""You are an expert code repository analyst.
Based on the following introductory section from a repository's README file, provide a detailed (2-3 sentences) summary of the repository's main purpose or function.
Analyze the snippet for:
- The core problem the repository addresses.
- Key features or functionalities mentioned.
- Any explicit or implicit target users or use cases.

README Introduction Snippet:
---
{intro_text}
---

Detailed Purpose Summary:"""
                    state["repository_purpose_summary"] = call_llm_api(prompt, f"Summarize repository purpose from README intro snippet '{intro_text[:50]}...'")
    
    _add_status(state, f"Finished processing READMEs. Purpose: {state['repository_purpose_summary']}")
    state["current_processing_stage"] = "process_configs"
    return state

def process_config_files(state: AgentState) -> AgentState:
    """
    Processes identified general configuration files.

    This node iterates through configuration files, attempts to parse them using
    specific parsers (if available, e.g., for XML) or a generic config parser.
    It stores the parsed data and may extract key information like project names.

    Args:
        state: The current agent state, containing `local_repo_path` and `config_files`.

    Returns:
        The updated agent state with parsed configuration data.
    """
    if not state["local_repo_path"]: return state
    _add_status(state, f"Processing {len(state['config_files'])} general configuration files...")

    for file_rel_path in state["config_files"]:
        # Avoid re-parsing if it was handled as a dependency file already and parsed there
        # This logic might need refinement based on how identify_dependencies_from_file stores its raw parsed data
        is_dependency_file_too = file_rel_path in state["dependency_files"]
        
        # For now, let's assume parse_generic_config_file is safe to call even if identify_dependencies also read it.
        # Or, we can add a check: if file_rel_path in state["parsed_dependency_file_details"]... skip.
        
        full_path = _get_file_path(state, file_rel_path)
        _add_status(state, f"Parsing generic config: {file_rel_path}")
        
        # Try specific parsers first if known, fallback to generic
        parsed_data = None
        if Path(file_rel_path).suffix.lower() == ".xml": # Example for specific handling if needed
            # parsed_data = tools.parse_xml_config(full_path) # If you had such a tool
            pass 

        if not parsed_data:
            parsed_data = tools.parse_generic_config_file(full_path)

        if parsed_data.get("error"):
            # If it's a dependency file, error might be acceptable if dep parser handles it
            if not is_dependency_file_too or "Unknown config file format" in parsed_data["error"]:
                 _add_error(state, f"Error parsing config file {file_rel_path}: {parsed_data['error']}")
        else:
            state["parsed_config_data"].append(parsed_data)
            # Potentially extract key info from configs for summary, e.g., project name from pyproject.toml
            if file_rel_path.endswith("pyproject.toml") and parsed_data.get("data", {}).get("project", {}).get("name"):
                proj_name = parsed_data["data"]["project"]["name"]
                _add_status(state, f"Found project name in pyproject.toml: {proj_name}")
                # Could add to component list or refine purpose summary
    
    _add_status(state, f"Finished processing {len(state['parsed_config_data'])} general configs.")
    state["current_processing_stage"] = "extract_dependencies"
    return state

def extract_dependencies(state: AgentState) -> AgentState:
    """
    Extracts dependencies from identified dependency management files.

    This node iterates through files like `requirements.txt`, `package.json`, etc.,
    uses a tool to identify main and development dependencies, and stores them
    in the agent state. It also prepares the queue for the next stage: source
    code metadata extraction.

    Args:
        state: The current agent state, containing `local_repo_path` and `dependency_files`.

    Returns:
        The updated agent state with extracted dependencies and the initial
        `files_to_parse_queue` for source code.
    """
    if not state["local_repo_path"]: return state
    _add_status(state, f"Extracting dependencies from {len(state['dependency_files'])} files...")
    
    all_deps = []
    all_dev_deps = []

    for file_rel_path in state["dependency_files"]:
        full_path = _get_file_path(state, file_rel_path)
        _add_status(state, f"Identifying dependencies in: {file_rel_path}")
        dep_data = tools.identify_dependencies_from_file(full_path)
        
        if dep_data.get("error"):
            _add_error(state, f"Error identifying dependencies in {file_rel_path}: {dep_data['error']}")
        else:
            if dep_data.get("dependencies"):
                all_deps.extend(dep_data["dependencies"])
            if dep_data.get("dev_dependencies"):
                all_dev_deps.extend(dep_data["dev_dependencies"])
            _add_status(state, f"Found {len(dep_data.get('dependencies',[]))} main, {len(dep_data.get('dev_dependencies',[]))} dev deps in {file_rel_path}")

    state["extracted_dependencies"]["main"] = sorted(list(set(all_deps)))
    state["extracted_dependencies"]["dev"] = sorted(list(set(all_dev_deps)))
    
    _add_status(state, f"Total unique main dependencies: {len(state['extracted_dependencies']['main'])}")
    _add_status(state, f"Total unique dev dependencies: {len(state['extracted_dependencies']['dev'])}")
    
    # Prepare for metadata extraction
    state["files_to_parse_queue"] = state["source_code_files"][:] # Start with all source files
    state["current_processing_stage"] = "process_source_metadata"
    return state

def select_batch_for_metadata_extraction(state: AgentState) -> AgentState:
    """
    Selects a batch of source code files for metadata extraction.

    This node manages the iterative processing of source files. It increments
    the iteration count, checks for maximum iteration limits, filters out
    already processed files, and selects the next batch based on
    `MAX_FILES_TO_PROCESS_PER_BATCH`. If no files remain or max iterations
    are reached, it directs the flow towards report generation.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with the `files_to_parse_queue` set for the
        current batch, or an empty queue if processing should stop.
    """
    if not state["local_repo_path"]: return state
    
    state["iteration_count"] += 1
    if state["iteration_count"] > state["max_iterations"]:
        _add_warning(state, f"Max iterations ({state['max_iterations']}) reached for metadata extraction.")
        state["files_to_parse_queue"] = [] # Stop processing
        state["current_processing_stage"] = "generate_report" # Move to report
        return state

    # Filter out already processed files
    remaining_files = [
        f for f in state["source_code_files"] 
        if f not in state["processed_metadata_files"]
    ]
    
    if not remaining_files:
        _add_status(state, "No more source files to process for metadata.")
        state["files_to_parse_queue"] = []
        state["current_processing_stage"] = "generate_report" # All done, move to report
        return state

    batch_size = MAX_FILES_TO_PROCESS_PER_BATCH
    current_batch = remaining_files[:batch_size]
    state["files_to_parse_queue"] = current_batch
    
    _add_status(state, f"Iteration {state['iteration_count']}: Selected batch of {len(current_batch)} files for metadata extraction.")
    return state

def extract_metadata_from_batch(state: AgentState) -> AgentState:
    """
    Extracts metadata from the current batch of source code files.

    This node iterates through the files in `files_to_parse_queue`. For each file,
    it attempts to use a language-specific parser (e.g., for Python) or falls
    back to reading a generic content preview. Extracted metadata is stored,
    and a summary for each component (file) is generated using an LLM call.
    Processed files are tracked.

    Args:
        state: The current agent state, containing `files_to_parse_queue`.

    Returns:
        The updated agent state with extracted metadata, component summaries,
        and updated tracking of processed files.
    """
    if not state["local_repo_path"] or not state["files_to_parse_queue"]:
        _add_status(state, "Skipping metadata extraction: no files in queue or no repo path.")
        return state

    _add_status(state, f"Extracting metadata from {len(state['files_to_parse_queue'])} files in current batch...")
    state["processed_files_this_iteration"] = []

    for file_rel_path in state["files_to_parse_queue"]:
        full_path = _get_file_path(state, file_rel_path)
        file_ext = Path(file_rel_path).suffix.lower()
        parsed_data = None

        if file_ext == ".py":
            _add_status(state, f"Parsing Python file: {file_rel_path}")
            parsed_data = tools.parse_python_file(full_path)
        elif file_ext in [".md", ".rst", ".txt"]: # Other text files that might be docs
             _add_status(state, f"Parsing Markdown/text (as potential doc): {file_rel_path}")
             parsed_data = tools.parse_markdown_file(full_path)
        # Add elif for other languages: .java, .js, etc.
        # elif file_ext == ".js":
        #    parsed_data = tools.parse_javascript_file(full_path) # Requires a js parser tool
        else:
            _add_status(state, f"Reading generic file (preview): {file_rel_path}")
            # For unknown source files, just read a preview for now
            # The agent could later decide to parse these more deeply if needed
            parsed_data = tools.read_file_content(full_path, char_limit=FILE_PREVIEW_CHAR_LIMIT)
            parsed_data["type"] = "generic_preview" # Add type for summarizer

        if parsed_data:
            if parsed_data.get("error"):
                _add_error(state, f"Error processing file {file_rel_path} for metadata: {parsed_data['error']}")
            else:
                state["extracted_metadata"][file_rel_path] = parsed_data
                # Summarize this component based on its metadata
                component_summary = summarize_component_from_metadata(file_rel_path, parsed_data)
                if component_summary:
                    state["key_components_summary"].append(component_summary)
        
        state["processed_metadata_files"].append(file_rel_path)
        state["processed_files_this_iteration"].append(file_rel_path)
    
    _add_status(state, f"Finished metadata extraction for batch. Total components summarized: {len(state['key_components_summary'])}")
    # No explicit next stage here, will be decided by conditional edge
    return state

def summarize_component_from_metadata(file_path: str, metadata: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Generates a summary for a single component (file) based on its extracted metadata.

    This function takes the file path and its parsed metadata (e.g., docstrings,
    class/function names for source code; sections for markdown) and constructs
    a prompt for an LLM to summarize the component's role and functionality.

    Args:
        file_path: The relative path of the file in the repository.
        metadata: A dictionary containing parsed information from the file.
                  The structure depends on the file type (e.g., Python, Markdown).

    Returns:
        A dictionary containing the component's name, its generated summary,
        and the source file path, or None if no substantial metadata was found
        for summarization.
    """
    text_for_summary = ""
    component_name = Path(file_path).name
    context_notes = [] # To guide the LLM

    if metadata.get("type") == "generic_preview": # From read_file_content
        text_for_summary = metadata.get("content", "")
        context_notes.append(f"File: {file_path}. Content preview.")
    elif metadata.get("file_path") and metadata.get("module_docstring"): # Python file
        component_name = Path(metadata["file_path"]).name
        text_for_summary += f"Module: {component_name}\nDocstring: {metadata['module_docstring']}\n"
        context_notes.append(f"Python module: {component_name}")
        if metadata.get("classes"):
            for cls_data in metadata["classes"][:2]: # Limit classes for brevity
                text_for_summary += f"Class: {cls_data['name']}\nDocstring: {cls_data.get('docstring','')}\n"
                if cls_data.get("methods"):
                     text_for_summary += f"Methods: {', '.join([m['name'] for m in cls_data['methods'][:3]])}\n" # Limit methods
        if metadata.get("functions"):
             text_for_summary += f"Functions: {', '.join([f['name'] for f in metadata['functions'][:3]])}\n" # Limit functions
    elif metadata.get("file_path") and metadata.get("sections"): # Markdown file
        component_name = Path(metadata["file_path"]).name
        context_notes.append(f"Documentation file: {component_name}")
        if metadata.get("title"): text_for_summary += f"Title: {metadata['title']}\n"
        for section in metadata["sections"][:2]: # Limit sections
            text_for_summary += f"Section: {section['heading']}\nPreview: {section['content_preview']}\n"
    
    if not text_for_summary.strip():
        # print(f"No substantial metadata to summarize for {file_path}")
        return None

    # Truncate if too long before sending to LLM
    if len(text_for_summary) > LLM_SUMMARY_CHAR_LIMIT:
        text_for_summary = text_for_summary[:LLM_SUMMARY_CHAR_LIMIT-3] + "..."
    
    prompt = f"""You are an expert code repository analyst.
The following text contains extracted metadata (like docstrings, class/function names, content previews, or import statements) from a file named '{component_name}'.
Based on this information, provide a detailed (2-4 sentences) summary of this file's primary role, responsibilities, and functionality within the repository.
Consider and include:
- Its specific responsibilities or tasks it performs.
- Key inputs it might process and outputs it might produce (if discernible).
- How it might interact with other components or parts of a larger system.
- Any specific algorithms, data structures, design patterns, or notable technologies used or implied by the metadata.

Extracted Metadata for {component_name}:
---
{text_for_summary}
---

Detailed Role/Functionality Summary:"""
    summary = call_llm_api(prompt, f"Summarize role of component '{component_name}'")
    return {"name": component_name, "summary": summary, "source_file": file_path}


def _add_warning(state: AgentState, message: str) -> None:
    """
    Adds a warning message to the agent's state and prints it.

    Args:
        state: The current agent state.
        message: The warning message to add.
    """
    # Similar to _add_error, but for warnings
    state["error_messages"].append(f"WARNING: {message}") # Store warnings in error_messages for now
    print(f"WARNING: {message}")


def generate_final_report(state: AgentState) -> AgentState:
    """
    Generates the final comprehensive report about the repository.

    This node synthesizes all collected information:
    - Refines the repository purpose summary if it's still basic, using combined README content.
    - Compiles key components, dependencies, a directory tree snippet, and processing counts.
    - Includes any errors or warnings encountered during the process.
    The final report is stored in `state["final_report"]`.

    Args:
        state: The current agent state, containing all extracted and summarized data.

    Returns:
        The updated agent state with the `final_report`.
    """
    _add_status(state, "Generating final report...")
    
    # Synthesize overall purpose if still basic
    if state["repository_purpose_summary"] == "Not yet determined." and state["parsed_readme_data"]:
        # Try to build a better one from all README content
        full_readme_text = ""
        for rd_data in state["parsed_readme_data"]:
            if rd_data.get("title"): full_readme_text += rd_data["title"] + "\n"
            for sec in rd_data.get("sections", []):
                full_readme_text += sec.get("heading","") + "\n" + sec.get("content_preview","") + "\n\n"
        if full_readme_text.strip():
            readme_content_for_llm = full_readme_text[:LLM_SUMMARY_CHAR_LIMIT]
            prompt = f"""You are an expert code repository analyst.
The following text is a compilation of content from one or more README files in a repository.
Please synthesize a comprehensive (3-5 sentences) overall summary of the repository.
Based on this combined information, detail:
- The primary problem domain the repository addresses.
- The core solution or functionality it offers.
- Key features, capabilities, or modules highlighted.
- The intended audience or typical use cases, if mentioned.
- Any insights into its development status, maturity, or unique aspects if evident from the text.

Combined README Content:
---
{readme_content_for_llm}
---

Comprehensive Overall Repository Summary:"""
            state["repository_purpose_summary"] = call_llm_api(prompt, "Synthesize overall repository purpose from combined READMEs")

    report = {
        "repository_url_or_path": state["repo_url_or_path"],
        "repository_purpose": state["repository_purpose_summary"],
        "key_components": state["key_components_summary"],
        "dependencies": state["extracted_dependencies"],
        "directory_tree_snippet": state.get("directory_tree", [])[:20] + ["... (and more)"] if state.get("directory_tree") and len(state.get("directory_tree",[])) > 20 else state.get("directory_tree", []),
        "processed_files_count": {
            "readmes": len(state["readme_files"]),
            "configs": len(state["config_files"]),
            "dependency_files": len(state["dependency_files"]),
            "source_code_metadata_extracted": len(state["processed_metadata_files"]),
        },
        "errors_encountered": state["error_messages"],
        "status_log": state["status_messages"]
    }
    state["final_report"] = report
    _add_status(state, "Final report generated.")
    # print(json.dumps(report, indent=2)) # For debugging
    return state

# --- Conditional Edges ---

def should_continue_metadata_extraction(state: AgentState) -> Literal["extract_metadata_from_batch", "generate_final_report"]:
    """
    Determines the next step after selecting a batch for metadata extraction.

    This conditional edge function checks if the maximum iteration count for
    metadata extraction has been reached or if there are no more source files
    left to process.

    Args:
        state: The current agent state.

    Returns:
        "generate_final_report" if processing should stop.
        "extract_metadata_from_batch" if there are files in the queue (this logic is handled by the
        conditional edge definition in `build_graph` based on `state["files_to_parse_queue"]`).
        The direct return here guides the graph if iterations are maxed out or all files are done.
    """
    if state["iteration_count"] >= state["max_iterations"]:
        _add_warning(state, "Max iterations reached. Moving to report generation.")
        return "generate_final_report"
    
    remaining_files = [f for f in state["source_code_files"] if f not in state["processed_metadata_files"]]
    if not remaining_files:
        _add_status(state, "All source files processed for metadata. Moving to report generation.")
        return "generate_final_report"
    
    _add_status(state, f"Continuing metadata extraction. Iteration: {state['iteration_count'] +1}, Remaining files: {len(remaining_files)}")
    return "extract_metadata_from_batch" # This should actually point to the batch selection node

# --- Graph Assembly ---
def build_graph():
    """
    Builds and compiles the LangGraph workflow for the repository explainer agent.

    This function defines all the nodes (processing steps) and edges (transitions)
    of the state machine, including conditional edges for iterative processing.

    Returns:
        A compiled LangGraph application (the agent).
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initialize_repository_scan", initialize_repository_scan)
    workflow.add_node("process_readme_files", process_readme_files)
    workflow.add_node("process_config_files", process_config_files)
    workflow.add_node("extract_dependencies", extract_dependencies)
    
    # Metadata processing loop
    workflow.add_node("select_batch_for_metadata_extraction", select_batch_for_metadata_extraction)
    workflow.add_node("extract_metadata_from_batch", extract_metadata_from_batch)
    
    workflow.add_node("generate_final_report", generate_final_report)

    # Define edges
    workflow.set_entry_point("initialize_repository_scan")
    workflow.add_edge("initialize_repository_scan", "process_readme_files")
    workflow.add_edge("process_readme_files", "process_config_files")
    workflow.add_edge("process_config_files", "extract_dependencies")
    workflow.add_edge("extract_dependencies", "select_batch_for_metadata_extraction") # Start metadata loop

    # Conditional edge for metadata processing loop
    workflow.add_conditional_edges(
        "select_batch_for_metadata_extraction",
        # Based on the return value of select_batch_for_metadata_extraction (which sets files_to_parse_queue)
        # we decide if we go to extract_metadata_from_batch or generate_final_report
        lambda state: "extract_metadata_from_batch" if state.get("files_to_parse_queue") else "generate_final_report",
        {
            "extract_metadata_from_batch": "extract_metadata_from_batch",
            "generate_final_report": "generate_final_report"
        }
    )
    # After a batch is processed, go back to select the next batch
    workflow.add_edge("extract_metadata_from_batch", "select_batch_for_metadata_extraction")

    workflow.add_edge("generate_final_report", END)
    
    # Compile the graph
    # memory = SqliteSaver.from_conn_string(":memory:") # For in-memory persistence & debugging
    # app = workflow.compile(checkpointer=memory)
    app = workflow.compile()
    return app

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LangGraph Repository Explainer Agent")
    parser.add_argument("repo_url_or_path", help="Git URL of the repository or path to a local repository.")
    parser.add_argument("--is_local", action="store_true", help="Flag if the path provided is a local path.")
    args = parser.parse_args()

    print(f"Starting agent for: {args.repo_url_or_path}")

    agent_app = build_graph()

    initial_state = {
        "repo_url_or_path": args.repo_url_or_path,
        "is_local_path": args.is_local,
    }
    
    # For streaming results (intermediate states)
    # config = {"configurable": {"thread_id": "repo-explainer-thread"}} # Unique ID for the run
    # for event in agent_app.stream(initial_state, config=config, stream_mode="values"):
    #     final_state = event
    #     # print("\n--- Current State ---")
    #     # print(json.dumps(final_state, indent=2, default=str)) # Print current state if needed

    final_state = agent_app.invoke(initial_state)

    print("\n\n--- FINAL REPORT ---")
    if final_state.get("final_report"):
        print(json.dumps(final_state["final_report"], indent=2, default=str))
    else:
        print("No final report generated. Check errors:")
        print(json.dumps(final_state.get("error_messages", []), indent=2))

    # Clean up cloned repo if it was cloned
    if not args.is_local and os.path.exists(TEMP_REPO_DIR):
        print(f"\nCleaning up temporary directory: {TEMP_REPO_DIR}")
        # shutil.rmtree(TEMP_REPO_DIR) # Uncomment to automatically clean up
        print(f"Cleanup skipped for inspection. Manually remove: {TEMP_REPO_DIR}")

    print("\nAgent finished.")
