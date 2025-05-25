# Repo Explainer Agent

## Description

The Repo Explainer Agent is a tool designed to understand and explain code repositories. It analyzes the repository structure, code, and documentation to provide insights and answer user queries about the codebase.

## Features

*   **Repository Initialization:** Clones and sets up a given Git repository for analysis.
*   **Static Code Analysis:** Parses source code to identify definitions, dependencies, and relationships between different code components.
*   **Documentation Parsing:** Extracts information from documentation files (e.g., READMEs, wikis).
*   **High-Level Summarization:** Generates an initial overview of the repository's purpose and architecture.
*   **File Explanation:** Provides detailed explanations for specific files within the repository.
*   **Architecture Analysis:** Offers insights into the overall structure and design of the codebase.
*   **User Query Processing:** Understands and responds to natural language questions about the repository.
*   **Response Generation:** Generates coherent and informative answers based on the analysis.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/repo_explainer_agent.git
    cd repo_explainer_agent
    ```
2.  **Build the Docker image:**
    ```bash
    docker/build
    ```
    Alternatively, you can build it manually:
    ```bash
    docker build -t repo_explainer_agent .
    ```

## Usage

To run the Repo Explainer Agent, use the provided `docker/run` script or execute the Docker container directly.

**Example:**

To ask a question about a specific repository, you can run:

```bash
docker/run https://github.com/andreimatveyeu/jackmesh.git "Where is the class Port defined?"
```

This command will:
1.  Initialize the `jackmesh` repository.
2.  Analyze its contents.
3.  Attempt to answer the question "Where is the class Port defined?".

## Project Structure

The project is organized into the following main directories:

*   `src/`: Contains the core logic of the agent.
    *   `core/`: Core components like data models, parsers, and state management.
    *   `nodes/`: Different processing nodes responsible for specific tasks (e.g., static code analysis, documentation parsing).
    *   `utils/`: Utility functions used across the project.
    *   `graph.py`: Manages the graph representation of the repository and the flow of information between nodes.
*   `docker/`: Contains Docker-related files for building and running the agent in a containerized environment.
    *   `Dockerfile`: Defines the Docker image.
    *   `build`: Script to build the Docker image.
    *   `run`: Script to run the Docker container with specified arguments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
