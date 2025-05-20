import os
import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Attempt to import libraries, with fallbacks or notes for installation
try:
    import toml
except ImportError:
    toml = None  # Or raise an error, or provide instructions to install

try:
    from git import Repo, InvalidGitRepositoryError, NoSuchPathError
except ImportError:
    Repo = None
    InvalidGitRepositoryError = None
    NoSuchPathError = None


# --- File System & Git Tools ---

def clone_repo(git_url: str, local_path: str) -> Tuple[bool, str]:
    """
    Clones a Git repository to a local path.
    Returns a tuple (success_status, message_or_repo_path).
    """
    if Repo is None:
        return False, "GitPython library is not installed. Please install it: pip install GitPython"
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            # Check if it's already a git repo and matches the URL
            try:
                repo = Repo(local_path)
                if git_url in repo.remotes.origin.urls:
                    return True, f"Repository already exists at {local_path} and matches URL."
                else:
                    return False, f"Directory {local_path} exists but is not the correct repo or has no remote."
            except (InvalidGitRepositoryError, NoSuchPathError):
                 return False, f"Directory {local_path} exists but is not a valid Git repository."

        Repo.clone_from(git_url, local_path)
        return True, local_path
    except Exception as e:
        return False, f"Error cloning repository: {e}"

def get_directory_tree(repo_path: str, ignore_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Generates a string representation of the directory tree.
    Ignores common patterns like .git, __pycache__, node_modules, etc. by default.
    """
    if ignore_patterns is None:
        ignore_patterns = [
            r"^\.git$", r"^__pycache__$", r"^node_modules$", r"^target$", r"^build$",
            r"^\.DS_Store$", r"^.*\.pyc$", r"^.*\.swp$", r"^\.idea$", r"^\.vscode$"
        ]

    tree_lines = []
    repo_path_obj = Path(repo_path)

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Filter directories
        dirs[:] = [d for d in dirs if not any(re.match(pattern, d) for pattern in ignore_patterns)]
        
        level = root.replace(str(repo_path_obj), '').count(os.sep)
        indent = ' ' * 4 * level
        tree_lines.append(f"{indent}{Path(root).name}/")
        sub_indent = ' ' * 4 * (level + 1)
        
        # Filter files
        filtered_files = [f for f in files if not any(re.match(pattern, f) for pattern in ignore_patterns)]
        for f in filtered_files:
            tree_lines.append(f"{sub_indent}{f}")
            
    return tree_lines

# --- File Content Reading & Basic Parsing ---

def read_file_content(
    file_path: str, 
    max_lines: Optional[int] = None,
    char_limit: Optional[int] = None 
) -> Dict[str, Any]:
    """
    Reads content of a file.
    Can limit by lines or characters.
    Returns a dict with 'content', 'error', 'file_type', 'line_count', 'char_count'.
    """
    result = {
        "content": None,
        "error": None,
        "file_path": file_path,
        "file_type": Path(file_path).suffix.lower(),
        "line_count": 0,
        "char_count": 0,
        "preview": False
    }
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        result["line_count"] = len(lines)
        full_content = "".join(lines)
        result["char_count"] = len(full_content)

        if max_lines is not None and len(lines) > max_lines:
            result["content"] = "".join(lines[:max_lines])
            result["preview"] = True
        elif char_limit is not None and len(full_content) > char_limit:
            result["content"] = full_content[:char_limit]
            result["preview"] = True
        else:
            result["content"] = full_content
            
    except FileNotFoundError:
        result["error"] = "File not found."
    except Exception as e:
        result["error"] = f"Error reading file: {e}"
    return result

def search_in_file(file_path: str, patterns: List[str]) -> Dict[str, List[Dict[str, Union[int, str]]]]:
    """
    Searches for a list of regex patterns in a file.
    Returns a dictionary where keys are patterns and values are lists of matches (line number and matched line).
    """
    results: Dict[str, List[Dict[str, Union[int, str]]]] = {pattern: [] for pattern in patterns}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                for pattern in patterns:
                    if re.search(pattern, line):
                        results[pattern].append({"line_number": i + 1, "line_content": line.strip()})
    except FileNotFoundError:
        return {"error": [{"line_number": 0, "line_content": f"File not found: {file_path}"}]} # type: ignore
    except Exception as e:
        return {"error": [{"line_number": 0, "line_content": f"Error searching file: {e}"}]} # type: ignore
    return results

# --- Language/Format Specific Parsers ---

def parse_python_file(file_path: str) -> Dict[str, Any]:
    """
    Parses a Python file to extract module docstring, classes (name, docstring, methods),
    and functions (name, signature, docstring).
    Does NOT include full method/function bodies.
    """
    result: Dict[str, Any] = {
        "file_path": file_path,
        "module_docstring": None,
        "classes": [],
        "functions": [],
        "imports": [],
        "error": None
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)

        result["module_docstring"] = ast.get_docstring(tree)

        for node in tree.body:
            if isinstance(node, ast.Import):
                result["imports"].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module: # Can be None for relative imports like 'from . import foo'
                    result["imports"].append(node.module + "." + ", ".join([alias.name for alias in node.names]))
                else: # Relative import
                     result["imports"].append("." * node.level + ", ".join([alias.name for alias in node.names]))


            elif isinstance(node, ast.ClassDef):
                class_info: Dict[str, Any] = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "attributes": [] # Basic instance attributes from __init__
                }
                for item in node.body:
                    if isinstance(item, ast.FunctionDef): # Method
                        method_info = {
                            "name": item.name,
                            "signature": ast.unparse(item.args), # type: ignore
                            "docstring": ast.get_docstring(item),
                            "decorators": [ast.unparse(d) for d in item.decorator_list] # type: ignore
                        }
                        class_info["methods"].append(method_info)
                        # Look for self.attr = ... in __init__
                        if item.name == "__init__":
                            for stmt in item.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                            class_info["attributes"].append(target.attr)
                    elif isinstance(item, ast.AnnAssign) or isinstance(item, ast.Assign): # Class variables
                        # This can be complex; for now, just grab names if simple
                        targets = []
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            targets.append(item.target.id)
                        elif isinstance(item, ast.Assign):
                            for target_node in item.targets:
                                if isinstance(target_node, ast.Name):
                                    targets.append(target_node.id)
                        if targets:
                             class_info["attributes"].extend(targets)


                result["classes"].append(class_info)
            elif isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "signature": ast.unparse(item.args), # type: ignore
                    "docstring": ast.get_docstring(node),
                    "decorators": [ast.unparse(d) for d in node.decorator_list] # type: ignore
                }
                result["functions"].append(func_info)
    except FileNotFoundError:
        result["error"] = "File not found."
    except SyntaxError as e:
        result["error"] = f"Syntax error parsing Python file: {e}"
    except Exception as e:
        result["error"] = f"Error parsing Python file: {e}"
    return result

def parse_markdown_file(file_path: str, max_chars_per_section: int = 500) -> Dict[str, Any]:
    """
    Extracts headings, introductory paragraphs, lists, and example code blocks from Markdown.
    """
    result: Dict[str, Any] = {
        "file_path": file_path,
        "title": None,
        "sections": [], # List of {"heading": str, "content_preview": str, "code_blocks": List[str]}
        "links": [],
        "error": None
    }
    try:
        content_data = read_file_content(file_path)
        if content_data["error"]:
            result["error"] = content_data["error"]
            return result
        
        content = content_data["content"]
        if not content:
             result["error"] = "Empty file or read error."
             return result


        # Simple title extraction (first H1 or first line if no H1)
        title_match = re.search(r"^\s*#\s+(.+)", content, re.MULTILINE)
        if title_match:
            result["title"] = title_match.group(1).strip()
        else:
            result["title"] = content.split('\n', 1)[0].strip()
            if len(result["title"]) > 100 : result["title"] = result["title"][:100] + "..."


        # Regex for sections (headings) and code blocks
        # This is a simplified approach. A proper Markdown parser would be more robust.
        # Sections are defined by headings (e.g., ## My Section)
        # Code blocks are ```lang ... ``` or indented blocks

        # Find all headings
        headings = list(re.finditer(r"^(#+)\s+(.*)", content, re.MULTILINE))
        
        current_section_content = []
        current_heading = result["title"] if result["title"] else "Introduction"
        current_code_blocks = []

        def store_section(heading, text_content, code_blocks_list):
            preview = " ".join(text_content).strip()
            if len(preview) > max_chars_per_section:
                preview = preview[:max_chars_per_section] + "..."
            
            # Basic link extraction from preview
            for link_match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', preview):
                result["links"].append({"text": link_match.group(1), "url": link_match.group(2)})

            result["sections"].append({
                "heading": heading,
                "content_preview": preview,
                "code_blocks": code_blocks_list[:]
            })

        # Split content by code blocks first to handle them separately
        parts = re.split(r"(```[\s\S]*?```|`[^`]*`)", content) # Split by fenced code blocks and inline code
        
        text_buffer = ""
        for i, part in enumerate(parts):
            if i % 2 == 1: # This is a code block or inline code
                if part.startswith("```"):
                    lang_match = re.match(r"```(\w*)\n", part)
                    lang = lang_match.group(1) if lang_match else "unknown"
                    code = part.strip("`\n ")
                    if lang_match: code = code[len(lang):].strip() # remove lang specifier
                    current_code_blocks.append({"language": lang, "code": code[:max_chars_per_section*2] + ("..." if len(code) > max_chars_per_section*2 else "")}) # Limit code block size too
                else: # Inline code, add to text buffer
                    text_buffer += part
            else: # This is normal text
                text_buffer += part
        
        # Now process text_buffer for headings
        # This is still simplistic; a full parser would be better.
        # For now, we'll just take the first few lines as intro if no headings.
        if not headings and text_buffer.strip():
            store_section(current_heading, text_buffer.split('\n')[:20], current_code_blocks) # take up to 20 lines for intro
        elif text_buffer.strip(): # If there's text, assume it's part of the last section or intro
            # This part is tricky without a proper parser state machine.
            # We'll associate all found code blocks with the first section for now.
            # A more advanced approach would interleave text and code block parsing.
            if result["sections"]:
                 result["sections"][0]["code_blocks"].extend(current_code_blocks)
                 result["sections"][0]["content_preview"] += "\n" + text_buffer[:max_chars_per_section] + ("..." if len(text_buffer) > max_chars_per_section else "")
            else: # No sections yet, treat as intro
                store_section(current_heading, text_buffer.split('\n')[:20], current_code_blocks)


    except FileNotFoundError:
        result["error"] = "File not found."
    except Exception as e:
        result["error"] = f"Error parsing Markdown file: {e}"
    return result


def identify_dependencies_from_file(file_path: str) -> Dict[str, Any]:
    """
    Identifies dependencies from known dependency files.
    Supports: requirements.txt, pyproject.toml (poetry & PDM/PEP 621), package.json.
    """
    file_name = Path(file_path).name.lower()
    result: Dict[str, Any] = {"file_path": file_path, "dependencies": [], "dev_dependencies": [], "error": None, "type": "unknown"}

    try:
        content_data = read_file_content(file_path)
        if content_data["error"]:
            result["error"] = content_data["error"]
            return result
        content = content_data["content"]
        if not content:
            result["error"] = "Empty dependency file."
            return result

        if file_name == "requirements.txt":
            result["type"] = "requirements.txt"
            deps = [
                line.strip().split("==")[0].split(">=")[0].split("<=")[0].split("!=")[0].split("~=")[0]
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]
            result["dependencies"] = list(set(d for d in deps if d)) # Unique, non-empty
        
        elif file_name == "pyproject.toml":
            result["type"] = "pyproject.toml"
            if toml is None:
                result["error"] = "toml library not installed. Cannot parse pyproject.toml."
                return result
            data = toml.loads(content)
            
            # PEP 621 dependencies (standard)
            if "project" in data and "dependencies" in data["project"]:
                result["dependencies"].extend(list(data["project"]["dependencies"]))
            if "project" in data and "optional-dependencies" in data["project"]:
                for group_name, deps_list in data["project"]["optional-dependencies"].items():
                    # Assuming all optional are dev for now, can be refined
                    result["dev_dependencies"].extend(deps_list)

            # Poetry
            if "tool" in data and "poetry" in data["tool"]:
                if "dependencies" in data["tool"]["poetry"]:
                    # Poetry includes python version, remove it
                    result["dependencies"].extend([dep for dep in data["tool"]["poetry"]["dependencies"] if dep.lower() != "python"])
                if "dev-dependencies" in data["tool"]["poetry"]: # Poetry pre 1.2
                     result["dev_dependencies"].extend([dep for dep in data["tool"]["poetry"]["dev-dependencies"]])
                if "group" in data["tool"]["poetry"] and "dev" in data["tool"]["poetry"]["group"] and \
                   "dependencies" in data["tool"]["poetry"]["group"]["dev"]: # Poetry 1.2+
                    result["dev_dependencies"].extend([dep for dep in data["tool"]["poetry"]["group"]["dev"]["dependencies"]])
            
            # PDM
            if "tool" in data and "pdm" in data["tool"]:
                if "dependencies" in data["tool"]["pdm"]:
                     result["dependencies"].extend([dep.split("=")[0].strip() for dep in data["tool"]["pdm"]["dependencies"]]) # PDM might have versions
                if "dev-dependencies" in data["tool"]["pdm"] and "dev" in data["tool"]["pdm"]["dev-dependencies"]: # PDM groups
                     result["dev_dependencies"].extend([dep.split("=")[0].strip() for dep in data["tool"]["pdm"]["dev-dependencies"]["dev"]])
            
            # Clean up duplicates
            result["dependencies"] = list(set(dep.split("[")[0].strip() for dep in result["dependencies"])) # Remove extras like [dev]
            result["dev_dependencies"] = list(set(dep.split("[")[0].strip() for dep in result["dev_dependencies"]))


        elif file_name == "package.json":
            result["type"] = "package.json"
            data = json.loads(content)
            if "dependencies" in data:
                result["dependencies"] = list(data["dependencies"].keys())
            if "devDependencies" in data:
                result["dev_dependencies"] = list(data["devDependencies"].keys())
        
        elif file_name == "setup.py" or file_name == "setup.cfg":
            result["type"] = file_name
            # Simplified regex search for install_requires in setup.py
            # This is fragile; ast parsing is more robust but complex for setup.py
            if file_name == "setup.py":
                matches = re.findall(r"install_requires\s*=\s*\[([^\]]*)\]", content, re.DOTALL)
                deps = []
                for match_group in matches:
                    deps.extend([dep.strip().strip("'\"") for dep in match_group.split(',') if dep.strip()])
                result["dependencies"] = list(set(d.split("==")[0].split(">=")[0] for d in deps if d))
            elif file_name == "setup.cfg":
                # configparser could be used here for more robustness
                if "[options]" in content:
                    matches = re.findall(r"install_requires\s*=\s*([\s\S]*?)(?=\n\w|\Z)", content) # Read until next key or EOF
                    deps = []
                    for m in matches:
                        deps.extend([line.strip() for line in m.splitlines() if line.strip() and not line.strip().startswith("#")])
                    result["dependencies"] = list(set(d.split("==")[0].split(">=")[0] for d in deps if d))


        else:
            result["error"] = f"Unsupported dependency file type: {file_name}"

    except json.JSONDecodeError:
        result["error"] = f"Invalid JSON in {file_path}"
    except Exception as e:
        result["error"] = f"Error parsing dependency file {file_path}: {e}"
    
    # Remove any empty strings that might have crept in
    result["dependencies"] = [d for d in result["dependencies"] if d]
    result["dev_dependencies"] = [d for d in result["dev_dependencies"] if d]
    return result

def parse_generic_config_file(file_path: str) -> Dict[str, Any]:
    """
    Attempts to parse common config file formats (JSON, TOML, YAML - if PyYAML installed).
    Returns a dictionary of the parsed data or an error.
    """
    result = {"file_path": file_path, "data": None, "error": None, "format": "unknown"}
    
    content_data = read_file_content(file_path)
    if content_data["error"]:
        result["error"] = content_data["error"]
        return result
    content = content_data["content"]
    if not content:
        result["error"] = "Empty config file."
        return result

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == ".json":
            result["data"] = json.loads(content)
            result["format"] = "json"
        elif file_ext == ".toml":
            if toml:
                result["data"] = toml.loads(content)
                result["format"] = "toml"
            else:
                result["error"] = "toml library not installed."
        elif file_ext in [".yaml", ".yml"]:
            try:
                import yaml # type: ignore
                result["data"] = yaml.safe_load(content)
                result["format"] = "yaml"
            except ImportError:
                result["error"] = "PyYAML library not installed. Cannot parse YAML."
            except yaml.YAMLError as e:
                 result["error"] = f"YAML parsing error: {e}"
        # Add INI parsing with configparser if needed
        # elif file_ext == ".ini":
        #     import configparser
        #     parser = configparser.ConfigParser()
        #     parser.read_string(content)
        #     result["data"] = {section: dict(parser.items(section)) for section in parser.sections()}
        #     result["format"] = "ini"
        else:
            # Try to infer by content (e.g. if it looks like JSON)
            try:
                result["data"] = json.loads(content)
                result["format"] = "json (inferred)"
            except json.JSONDecodeError:
                if toml:
                    try:
                        result["data"] = toml.loads(content)
                        result["format"] = "toml (inferred)"
                    except toml.TomlDecodeError:
                        result["error"] = "Unknown config file format or parse error."
                else:
                    result["error"] = "Unknown config file format or parse error (toml not available for inference)."
        return result
    except Exception as e:
        result["error"] = f"Error parsing generic config file: {e}"
        return result

if __name__ == '__main__':
    # Example Usage (for testing tools)
    # Create dummy files and directories for testing
    test_repo_path = "test_repo"
    if not os.path.exists(test_repo_path):
        os.makedirs(test_repo_path)
    
    with open(os.path.join(test_repo_path, "README.md"), "w") as f:
        f.write("# Test Project\n\nThis is a test project.\n\n```python\nprint('hello')\n```\n\nLink: [Example](http://example.com)")

    with open(os.path.join(test_repo_path, "main.py"), "w") as f:
        f.write("\"\"\"Module docstring.\"\"\"\nimport os\n\nclass MyClass:\n    \"\"\"Class docstring.\"\"\"\n    def __init__(self, x):\n        self.x = x\n\n    def greet(self, name: str) -> str:\n        \"\"\"Method docstring.\"\"\"\n        return f\"Hello, {name}! I have {self.x}\"\n\ndef top_level_func(y):\n    \"\"\"Function docstring.\"\"\"\n    return y * 2")

    with open(os.path.join(test_repo_path, "requirements.txt"), "w") as f:
        f.write("numpy==1.20.0\npandas\n# This is a comment\nrequests>=2.0")

    with open(os.path.join(test_repo_path, "pyproject.toml"), "w") as f:
        f.write("[tool.poetry.dependencies]\npython = \"^3.8\"\nfastapi = \"^0.70.0\"\n\n[tool.poetry.dev-dependencies]\npytest = \"^6.0\"\n\n[project.dependencies]\nflask = \"1.1\"")
    
    os.makedirs(os.path.join(test_repo_path, ".git"), exist_ok=True) # Ignored dir
    os.makedirs(os.path.join(test_repo_path, "src"), exist_ok=True)
    with open(os.path.join(test_repo_path, "src", "utils.py"), "w") as f:
        f.write("# Util functions")

    print("--- Directory Tree ---")
    tree = get_directory_tree(test_repo_path)
    for line in tree:
        print(line)

    print("\n--- Read README.md (partial) ---")
    readme_data = read_file_content(os.path.join(test_repo_path, "README.md"), max_lines=3)
    print(json.dumps(readme_data, indent=2))
    
    print("\n--- Parse main.py ---")
    py_data = parse_python_file(os.path.join(test_repo_path, "main.py"))
    print(json.dumps(py_data, indent=2))

    print("\n--- Search in main.py for 'Hello' ---")
    search_results = search_in_file(os.path.join(test_repo_path, "main.py"), ["Hello", "nonexistent"])
    print(json.dumps(search_results, indent=2))

    print("\n--- Identify Dependencies (requirements.txt) ---")
    req_deps = identify_dependencies_from_file(os.path.join(test_repo_path, "requirements.txt"))
    print(json.dumps(req_deps, indent=2))

    print("\n--- Identify Dependencies (pyproject.toml) ---")
    toml_deps = identify_dependencies_from_file(os.path.join(test_repo_path, "pyproject.toml"))
    print(json.dumps(toml_deps, indent=2))
    
    print("\n--- Parse Markdown (README.md) ---")
    md_data = parse_markdown_file(os.path.join(test_repo_path, "README.md"))
    print(json.dumps(md_data, indent=2))

    # Clean up test files
    # import shutil
    # shutil.rmtree(test_repo_path)
    print(f"\nNote: Test files/dirs created in '{test_repo_path}'. Remove manually or uncomment cleanup.")
