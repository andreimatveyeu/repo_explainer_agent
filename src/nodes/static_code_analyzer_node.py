import os
from typing import Dict, Any, List, Optional, Type

from src.core.state import RepoExplainerState
from src.core.models import AbstractCodeKnowledgeBase, CodeEntity
from src.core.parsers import ILanguageParser, PythonParser

# A registry for available language parsers
# In a more dynamic system, this could be populated via plugins or configuration
LANGUAGE_PARSERS: Dict[str, Type[ILanguageParser]] = {
    "python": PythonParser,
    # Add other parsers here as they are implemented, e.g.:
    # "javascript": JavaScriptParser,
}

def get_parser_for_file(file_path: str) -> Optional[ILanguageParser]:
    """
    Determines the appropriate parser for a given file based on its extension.
    """
    _, extension = os.path.splitext(file_path)
    if not extension:
        return None

    for lang, parser_class in LANGUAGE_PARSERS.items():
        parser_instance = parser_class() # Create an instance to call get_supported_extensions
        if extension.lower() in parser_instance.get_supported_extensions():
            return parser_instance # Return a new instance for parsing
    return None


def static_code_analyzer(state: RepoExplainerState) -> RepoExplainerState:
    """
    Performs static code analysis on the repository specified by `local_repo_path`.
    It iterates through files, invokes appropriate parsers, and populates
    the `abstract_code_kb` in the state.
    """
    print("--- Running Node: static_code_analyzer ---")
    local_repo_path: Optional[str] = state.get("local_repo_path")
    updated_state: Dict[str, Any] = {}

    if not local_repo_path or not os.path.isdir(local_repo_path):
        message = f"Invalid or missing local_repo_path: {local_repo_path}. Cannot perform static analysis."
        print(f"Error: {message}")
        updated_state["error_message"] = message
        updated_state["abstract_code_kb"] = AbstractCodeKnowledgeBase() # Empty KB
        return {**state, **updated_state}

    code_kb = AbstractCodeKnowledgeBase()
    repo_config = state.get("repo_config", {})
    detected_languages_in_repo = set(repo_config.get("detected_languages", []))

    # Store entities parsed per language for potential batch dependency resolution
    parsed_entities_by_lang: Dict[str, List[CodeEntity]] = {}

    print(f"Starting static analysis for repository: {local_repo_path}")
    for root, _, files in os.walk(local_repo_path):
        # Basic exclusion for .git directory and other common non-code dirs
        # This should be made more configurable (e.g., via .gitignore or config file)
        if ".git" in root.split(os.sep) or \
           "node_modules" in root.split(os.sep) or \
           "__pycache__" in root.split(os.sep) or \
           "target" in root.split(os.sep) or \
           "build" in root.split(os.sep) or \
           "dist" in root.split(os.sep): # Add more common build/dependency dirs
            continue

        for file_name in files:
            file_path = os.path.join(root, file_name)
            parser = get_parser_for_file(file_path)

            if parser:
                language = "" # Determine language from parser if possible, or hardcode for now
                for lang_name, p_class in LANGUAGE_PARSERS.items():
                    if isinstance(parser, p_class):
                        language = lang_name
                        break
                
                if language and language not in detected_languages_in_repo:
                    detected_languages_in_repo.add(language)
                
                print(f"Parsing ({language or 'unknown language'}): {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_entities = parser.parse_file_content(file_path, content)
                    if not file_entities: # Ensure at least a file entity is created, even for empty/unparseable
                         # This case should ideally be handled by the parser itself,
                         # creating a basic 'file' CodeEntity.
                        print(f"Warning: Parser returned no entities for {file_path}")
                        # Potentially create a default file entity here if parser doesn't
                        
                    for entity in file_entities:
                        code_kb.add_entity(entity)
                    
                    if language:
                        if language not in parsed_entities_by_lang:
                            parsed_entities_by_lang[language] = []
                        parsed_entities_by_lang[language].extend(file_entities)

                except Exception as e:
                    print(f"Error parsing file {file_path}: {e}")
                    # Optionally, create a CodeEntity representing the errored file
                    error_entity_id = f"{file_path}::ERROR"
                    error_entity = CodeEntity(
                        id=error_entity_id,
                        entity_type="file", # Treat as a file entity with error metadata
                        name=file_name,
                        qualified_name=file_path,
                        language=language or "unknown",
                        filepath=file_path,
                        start_line=1,
                        end_line=1, # Placeholder
                        summary=f"Failed to parse file.",
                        metadata={"parsing_error": str(e)}
                    )
                    code_kb.add_entity(error_entity)
            # else:
            #     print(f"No parser found for: {file_path}")


    # Optional: Second pass for dependency resolution if parsers support it
    # This allows parsers to link dependencies across the entire KB once all entities are initially parsed.
    print("Attempting dependency resolution pass...")
    for lang, parser_class in LANGUAGE_PARSERS.items():
        if lang in parsed_entities_by_lang and parsed_entities_by_lang[lang]:
            parser_instance = parser_class() # Get a fresh instance
            print(f"Resolving dependencies for {lang}...")
            # The parser's resolve_dependencies might modify entities in place within code_kb.entities
            # by looking them up in the provided map.
            parser_instance.resolve_dependencies(parsed_entities_by_lang[lang], code_kb.entities)


    repo_config["detected_languages"] = list(detected_languages_in_repo)
    code_kb.detected_languages = list(detected_languages_in_repo) # Also store in KB itself

    updated_state["abstract_code_kb"] = code_kb
    updated_state["repo_config"] = repo_config
    updated_state["error_message"] = None # Clear previous errors if analysis ran
    print(f"Static analysis completed. KB: {code_kb}")
    
    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage (for testing the node directly)
    
    # Create a dummy repo structure for testing
    test_repo_dir = "./temp_analyzer_test_repo"
    os.makedirs(os.path.join(test_repo_dir, "subdir"), exist_ok=True)

    py_file_content = """
# This is a test Python file
import os
import sys

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        '''Greets the user.'''
        return f"Hello, {self.name}!"

def top_level_function(x, y):
    # A simple function
    return x + y

CONSTANT_VAR = 123
"""
    with open(os.path.join(test_repo_dir, "main.py"), "w") as f:
        f.write(py_file_content)
    with open(os.path.join(test_repo_dir, "subdir", "utils.py"), "w") as f:
        f.write("def utility_func():\n    return 'utility'")
    with open(os.path.join(test_repo_dir, "README.md"), "w") as f:
        f.write("# Test Readme\nThis is a test.")

    initial_state: RepoExplainerState = {
        "user_query": "Analyze this test repo.",
        "local_repo_path": test_repo_dir,
        "repo_config": {} # Start with empty repo_config
    }

    print(f"\nTesting static_code_analyzer with repo: {test_repo_dir}")
    result_state = static_code_analyzer(initial_state)

    print(f"\nResulting state's error_message: {result_state.get('error_message')}")
    
    kb = result_state.get("abstract_code_kb")
    if kb:
        print(f"Knowledge Base: {kb}")
        print(f"Detected languages in KB: {kb.detected_languages}")
        print(f"Detected languages in repo_config: {result_state.get('repo_config', {}).get('detected_languages')}")
        
        assert "python" in kb.detected_languages
        
        # Check for some expected entities
        main_py_file_entities = kb.get_file_entities(os.path.join(test_repo_dir, "main.py"))
        assert len(main_py_file_entities) > 0 # File entity itself + others
        
        found_myclass = False
        found_greet_method = False
        found_toplevel_func = False
        
        for entity_id, entity_obj in kb.entities.items():
            if entity_obj.name == "MyClass" and entity_obj.entity_type == "class":
                found_myclass = True
                print(f"Found MyClass: {entity_obj.id}, children: {entity_obj.children_ids}")
                # Check if greet method is a child
                for child_id in entity_obj.children_ids:
                    child_entity = kb.get_entity(child_id)
                    if child_entity and child_entity.name == "greet" and child_entity.entity_type == "method":
                        found_greet_method = True
                        print(f"Found greet method: {child_entity.id}")
                        break
            if entity_obj.name == "top_level_function" and entity_obj.entity_type == "function":
                found_toplevel_func = True
                print(f"Found top_level_function: {entity_obj.id}")

        assert found_myclass
        assert found_greet_method
        assert found_toplevel_func
        
        # Test dependency resolution (basic import os/sys)
        import_entities = kb.get_entities_by_type("import_statement", language="python")
        print(f"\nFound {len(import_entities)} import statements.")
        for imp_entity in import_entities:
            print(f"Import: {imp_entity.name}, Dependencies: {imp_entity.dependencies}")
            # In this basic setup, 'os' and 'sys' won't resolve to entities in our small test repo,
            # so dependencies list for these might be empty unless we mock stdlib entities.
            # The test for resolve_dependencies in PythonParser is more about structure.

    else:
        print("No AbstractCodeKnowledgeBase found in result state.")

    # Cleanup
    if os.path.exists(test_repo_dir):
        import shutil
        shutil.rmtree(test_repo_dir)
    
    print("\nstatic_code_analyzer node tests completed.")
