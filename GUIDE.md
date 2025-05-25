# Implementation Guide: Repository Explainer Agent

**1. Introduction & Core Principles**

*   **1.1. Project Vision:**
    *   To create an intelligent, interactive, and language-agnostic agent capable of providing deep, contextual explanations of software repositories.
    *   The agent will go beyond simple summarization, offering insights into code structure, dependencies, architecture, and purpose.
*   **1.2. Core Architectural Principles:**
    *   **Modularity:** Components (LangGraph nodes, parsers) should be well-defined and loosely coupled.
    *   **Extensibility:** The architecture must easily accommodate new programming languages, analysis tools, and agent capabilities.
    *   **Language Agnosticism (at the core):** Reasoning and explanation logic should operate on a standardized, abstract representation of code, independent of specific language syntax.
    *   **Agentic Behavior:** The system should understand user intent, plan actions, maintain conversational context, and interact naturally.
    *   **Accuracy & Depth:** Explanations should be grounded in thorough code analysis.
    *   **User-Centricity:** The agent's interactions and outputs should be clear, helpful, and adaptable to user needs.
*   **1.3. Key Technologies (Initial Stack):**
    *   **Orchestration:** LangGraph
    *   **Core Logic:** Python
    *   **LLMs:** (Specify preferred models/APIs, e.g., OpenAI GPT-series, Anthropic Claude, or open-source models via Hugging Face)
    *   **State Management:** Pydantic for `RepoExplainerState` and `CodeEntity`.
    *   **Language Parsing:** Standard libraries (e.g., Python `ast`) and potentially external tools/LSPs where necessary.

**2. Phase 1: Foundational Infrastructure & Single Language MVP (e.g., Python)**

    This phase focuses on building the core data structures, the initial LangGraph flow, and support for one primary language to validate the architecture.

*   **2.1. `RepoExplainerState` Detailed Definition:**
    *   Finalize all fields using `TypedDict` and `NotRequired`.
    *   Emphasize `abstract_code_kb: AbstractCodeKnowledgeBase` and `explanation_history: List[str]`.
    *   Define `repo_config: Dict[str, Any]` for storing detected languages, project type, etc.
*   **2.2. `AbstractCodeKnowledgeBase` (ACKB) Implementation:**
    *   **2.2.1. `CodeEntity` Pydantic Model:**
        *   Define all attributes as discussed (id, entity_type, name, qualified_name, language, filepath, start/end_line, raw_text_snippet, summary, docstring, dependencies, dependents, parent_id, children_ids, metadata).
        *   Provide clear descriptions and examples for each attribute.
        *   `id` generation strategy: Ensure uniqueness (e.g., `filepath::qualified_name` or hash).
    *   **2.2.2. `ACKB` Class:**
        *   Internal storage: `entities: Dict[str, CodeEntity]`.
        *   Methods: `add_entity`, `get_entity`, `find_entities` (by name, type, language), `get_dependencies`, `get_dependents`, `get_file_entities`, `get_entities_by_type`, etc.
        *   Consider initial in-memory implementation.
*   **2.3. Language Adapter Framework:**
    *   **2.3.1. `ILanguageParser` (Abstract Base Class / Interface):**
        *   `detect(file_path: str, file_content: str) -> bool`: Can this parser handle this file?
        *   `parse(file_path: str, file_content: str) -> List[CodeEntity]`: Parses a single file into `CodeEntity` objects. Focus on structure, not full semantic understanding initially.
        *   `resolve_dependencies(entities: List[CodeEntity], all_repo_entities_accessor: Callable[[str], Optional[CodeEntity]]) -> None`: (Optional, can be a post-processing step) Updates `dependencies`/`dependents` fields in the provided entities.
    *   **2.3.2. `PythonParser(ILanguageParser)` Implementation:**
        *   Use Python's `ast` module.
        *   Map Python AST nodes (ClassDef, FunctionDef, ImportFrom, etc.) to `CodeEntity` attributes.
        *   Extract docstrings into `CodeEntity.docstring`.
        *   Initial dependency identification: Populate `CodeEntity.dependencies` based on import statements.
*   **2.4. Core LangGraph Nodes (Initial Implementation):**
    *   **`initialize_repository`:** Clones/accesses repo, lists files. Output: `local_repo_path`.
    *   **`static_code_analyzer`:**
        *   Detects languages (initially, just Python via file extensions).
        *   Invokes `PythonParser` for `.py` files.
        *   Populates `abstract_code_kb` with `CodeEntity` objects.
        *   (Optional) Calls a dependency resolution step for the `abstract_code_kb`.
    *   **`documentation_parser`:** Parses `README.md` and other specified doc files. Output: `parsed_documentation: Dict[str, str]`.
    *   **`initial_high_level_summarizer`:** Uses LLM with `abstract_code_kb` (file list, main entities from Python) and `parsed_documentation` to create `overall_summary`.
    *   **`user_query_parser` (Basic):**
        *   Input: `user_query`, `overall_summary`.
        *   Action: LLM to identify simple intents ("explain file `foo.py`", "what is this repo about?") and target file paths.
        *   Output: `parsed_query_intent`, `target_entity` (e.g., filepath).
    *   **`file_explainer_node`:**
        *   Input: `target_entity` (filepath), `abstract_code_kb`.
        *   Action: Retrieve `CodeEntity` objects for the file. Use LLM to generate an explanation based on these entities and their raw code snippets/docstrings.
        *   Output: `generated_explanation`.
    *   **`response_generator_node`:** Formats `generated_explanation` for the user.
*   **2.5. Initial Graph Construction:**
    *   Define the LangGraph with the above nodes.
    *   Edges: Linear flow for ingestion (`initialize_repository` -> `static_code_analyzer` -> `documentation_parser` -> `initial_high_level_summarizer`).
    *   Loop: `initial_high_level_summarizer` -> `user_query_parser` -> (conditional based on intent) -> `file_explainer_node` -> `response_generator_node` -> `user_query_parser`.

**3. Phase 2: Advancing Agentic Capabilities & Query Richness**

    Build upon the MVP to make the agent more interactive, intelligent, and capable of handling complex queries.

*   **3.1. Enhanced `user_query_parser`:**
    *   **Intent Recognition:** Broader range of intents (explain relationship, explain architecture, find usages, compare entities).
    *   **Entity Disambiguation:** Use `abstract_code_kb` and `explanation_history` to resolve ambiguous entity names.
    *   **Mini-Plan Generation:** For complex queries, the LLM should output a sequence of actions/nodes to invoke.
    *   **Knowledge Gap Identification:** Determine if more analysis or tool use is needed.
    *   **Clarification Questions:** If query is ambiguous, formulate a question back to the user.
*   **3.2. New Explainer Nodes:**
    *   **`relationship_explainer_node`:**
        *   Input: `entity_ids` (from `abstract_code_kb`), `relationship_type_query`, `abstract_code_kb`.
        *   Action: Uses `ACKB` methods. LLM synthesizes explanation.
        *   Output: `generated_explanation`, `visualizations` (e.g., DOT language for simple graphs).
    *   **`architectural_overview_node`:**
        *   Input: `abstract_code_kb`, `overall_summary`.
        *   Action: LLM identifies patterns from high-level `CodeEntity` objects and their dependencies.
*   **3.3. `tool_executor_node` and Basic Tools:**
    *   Define a generic tool interface (e.g., `name`, `description`, `input_schema`, `execute_method`).
    *   Implement a "code_search_tool" (e.g., regex-based or using `git grep`).
    *   The `user_query_parser` can decide to call tools via `tool_calls` in the state.
*   **3.4. Context & Conversation Management:**
    *   Rigorously use `explanation_history` in `user_query_parser` and `response_generator_node`.
    *   Maintain `current_focus_path` or `current_focus_entity_id` in the state.
*   **3.5. Robust Error Handling:**
    *   Implement an `error_handler_node` in LangGraph.
    *   Nodes should catch their own exceptions and update `error_message` in the state.
    *   `response_generator_node` presents errors gracefully.

**4. Phase 3: Extensibility, Advanced Analysis & Polish**

    Focus on adding more languages, deeper analysis capabilities, and refining the user experience.

*   **4.1. Adding a Second Language (e.g., JavaScript):**
    *   Implement `JavaScriptParser(ILanguageParser)` (e.g., using `esprima-python` or a similar library).
    *   Map JS AST to `CodeEntity` objects.
    *   Register the parser with `static_code_analyzer`.
    *   Thoroughly test with JavaScript repositories.
*   **4.2. `commit_history_analyzer` Node:**
    *   Integrate `GitPython` or use `git` CLI commands.
    *   Extract commit messages.
    *   LLM to summarize changes relevant to a specific file/entity or general project evolution.
*   **4.3. Vector Database Integration (Semantic Search - Highly Recommended):**
    *   Choose a VDB (e.g., ChromaDB, FAISS, Weaviate).
    *   **`embedding_generator_node`:**
        *   Embed `CodeEntity` summaries, docstrings, and `parsed_documentation`.
        *   Store embeddings in VDB.
    *   **`semantic_search_node`:**
        *   Embed user query.
        *   Retrieve relevant code/doc chunks from VDB.
        *   This retrieved context can be fed into other explainer LLMs.
*   **4.4. Visualization Enhancements:**
    *   Generate more complex DOT outputs for dependency graphs, call graphs (if feasible).
    *   Consider if/how these can be rendered if the agent has a UI.
*   **4.5. (Optional) `dynamic_code_analyzer_node`:**
    *   For on-demand, deeper analysis of specific code sections if initial static analysis is insufficient (e.g., detailed call graph for one function). This would update the `abstract_code_kb`.

**5. Development, Testing & Operational Guidelines**

*   **5.1. LLM Prompt Engineering:**
    *   Maintain a version-controlled library of prompts for each LLM-driven node.
    *   Emphasize clear instructions, few-shot examples, and role-playing.
    *   Iteratively test and refine prompts.
*   **5.2. Testing Strategy:**
    *   **Unit Tests:** For individual parsers, `ACKB` methods, and helper functions.
    *   **Node Tests:** Test each LangGraph node in isolation with mocked state inputs.
    *   **Integration Tests:** Test full LangGraph flows with small, representative sample repositories (for each supported language).
    *   **Evaluation:**
        *   Define metrics for explanation quality (can be qualitative initially).
        *   Create a "golden dataset" of queries and expected explanation characteristics for key test repos.
*   **5.3. Configuration Management:**
    *   Externalize LLM API keys, model names, temperature settings.
    *   Configuration for parsers (e.g., paths to external tools if any).
*   **5.4. Logging & Debugging:**
    *   Implement comprehensive logging within each LangGraph node, recording key inputs and outputs.
    *   LangGraph's built-in debugging tools.
    *   Ability to easily inspect the `RepoExplainerState` at any step.
*   **5.5. Scalability & Performance:**
    *   Profile parsing times for large files/repositories.
    *   Optimize `ACKB` queries.
    *   Consider strategies for very large repos: on-demand parsing, indexing, focusing on sub-directories.
*   **5.6. Contribution Guide (for future development):**
    *   Clear instructions on how to add support for a new programming language (implementing `ILanguageParser`, registering it).
    *   Guidelines for adding new tools or agent capabilities.
    *   Coding standards and testing requirements.
