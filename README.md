# Repo Explainer Agent

A LangGraph-based AI agent designed to analyze the contents of a Git repository (or local directory) and generate a structured report summarizing its purpose, key components, and dependencies.

## Overview

The Repo Explainer Agent automates the initial exploration of a codebase. It performs the following main tasks:

1.  **Repository Access**: Clones a remote Git repository or accesses a specified local directory.
2.  **File Identification**: Scans the repository to categorize files into READMEs, configuration files, dependency manifests, and source code.
3.  **Content Parsing**:
    *   Extracts information from README files (titles, sections).
    *   Parses common configuration file formats (JSON, TOML, YAML).
    *   Identifies project dependencies from files like `requirements.txt`, `pyproject.toml`, and `package.json`.
    *   Analyzes source code (currently focused on Python) to extract metadata such as module/class/function docstrings, signatures, and imports.
4.  **Summarization**: Uses a mock LLM component to generate summaries for the repository's purpose and key code components based on extracted data.
5.  **Report Generation**: Compiles all gathered information into a comprehensive JSON report.

This agent is built using [LangGraph](https://langchain-ai.github.io/langgraph/) to manage the flow of operations as a state machine.

## Features

*   **Automated Repository Cloning**: Fetches remote repositories using Git.
*   **Local Repository Analysis**: Can analyze repositories already present on the local filesystem.
*   **Directory Tree Generation**: Provides a visual overview of the repository structure.
*   **File Type Categorization**: Intelligently identifies various important file types.
*   **Python Code Analysis**: Extracts docstrings, class structures, function signatures, and imports from Python files using Abstract Syntax Trees (AST).
*   **Markdown Parsing**: Processes README files to extract titles, section headings, content previews, and code blocks.
*   **Dependency Extraction**: Supports common Python and Node.js dependency files.
*   **Configuration File Parsing**: Handles JSON, TOML, and YAML (if PyYAML is installed).
*   **Iterative Processing**: Processes source code files in batches for metadata extraction.
*   **Mock LLM Summaries**: Demonstrates how an LLM could be used for summarizing code and repository purpose.
*   **Structured JSON Output**: Produces a detailed report for easy programmatic access or review.

## How it Works

The agent operates as a state graph, where each node represents a specific processing stage:

1.  **Initialization**: Clones or sets up the local repository path, identifies all files, and categorizes them.
2.  **README Processing**: Parses identified README files to understand the project's stated purpose and structure.
3.  **Configuration File Processing**: Parses general configuration files.
4.  **Dependency Extraction**: Analyzes dependency files to list project dependencies.
5.  **Source Code Metadata Extraction**: Iteratively processes source code files in batches:
    *   Selects a batch of files.
    *   Extracts metadata (e.g., docstrings, function/class definitions for Python files; content previews for others).
    *   Summarizes each processed component.
    *   Repeats until all source files are processed or a maximum iteration limit is reached.
6.  **Report Generation**: Consolidates all extracted information and summaries into a final JSON report.

## Core Components

The project is primarily structured into two Python files within the `repo_explainer_agent` package:

*   **`repo_explainer_agent.py`**: Contains the main agent logic, including the LangGraph state definitions, node implementations (processing steps), and graph assembly. It orchestrates the overall analysis process.
*   **`tools.py`**: Provides a collection of utility functions used by the agent. These include:
    *   Git operations (`clone_repo`).
    *   File system interactions (`get_directory_tree`, `read_file_content`).
    *   Language-specific parsers (`parse_python_file`, `parse_markdown_file`).
    *   Dependency and configuration file parsers (`identify_dependencies_from_file`, `parse_generic_config_file`).

## Usage

The agent is run from the command line.

**Prerequisites:**

*   Python 3.8+
*   Git installed on your system (for cloning remote repositories).
*   Required Python packages (see `Dependencies` section). You can typically install them using pip:
    ```bash
    pip install GitPython langgraph toml # PyYAML is optional for YAML config parsing
    ```

**Running the Agent:**

```bash
python -m repo_explainer_agent.repo_explainer_agent <repository_url_or_local_path> [--is_local]
```

**Arguments:**

*   `repository_url_or_local_path`:
    *   The URL of the Git repository to clone (e.g., `https://github.com/user/project.git`).
    *   The local file system path to an existing repository (e.g., `./my_local_project`).
*   `--is_local` (optional):
    *   A flag to indicate that the provided path is a local directory, not a URL to be cloned.

**Examples:**

1.  **Analyze a remote repository:**
    ```bash
    python -m repo_explainer_agent.repo_explainer_agent https://github.com/langchain-ai/langgraph
    ```

2.  **Analyze a local repository:**
    ```bash
    python -m repo_explainer_agent.repo_explainer_agent /path/to/your/local/repo --is_local
    ```

## Output

The agent prints status messages to the console during its operation and concludes by printing a **Final Report** in JSON format. This report includes:

*   `repository_url_or_path`: The input repository identifier.
*   `repository_purpose`: A summary of the repository's purpose (derived from READMEs or other content).
*   `key_components`: A list of summaries for key source files/components.
*   `dependencies`: Main and development dependencies found.
*   `directory_tree_snippet`: A preview of the repository's directory structure.
*   `processed_files_count`: Counts of different file types processed.
*   `errors_encountered`: A list of any errors that occurred during processing.
*   `status_log`: A log of status messages.

## Dependencies

*   **`langgraph`**: For building the stateful agent graph.
*   **`GitPython`**: For cloning Git repositories.
*   **`toml`**: For parsing `pyproject.toml` files (and other TOML files).
*   **`PyYAML`** (Optional): For parsing YAML configuration files. If not installed, YAML parsing will be skipped.

## Future Enhancements

*   Support for analyzing more programming languages (e.g., JavaScript, Java, Go).
*   Integration with a real Large Language Model (LLM) for more sophisticated summarization and analysis.
*   More detailed parsing of various configuration file types.
*   Ability to answer specific questions about the repository based on the analysis.
*   Enhanced error handling and reporting.
