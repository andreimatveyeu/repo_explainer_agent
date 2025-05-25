import ast
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Dict, Any

from .models import CodeEntity # Assuming models.py is in the same directory (core)

class ILanguageParser(ABC):
    """
    Abstract Base Class (Interface) for language-specific parsers.
    Each language parser will implement these methods to transform source code
    into a list of CodeEntity objects.
    """

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Returns a list of file extensions this parser supports (e.g., ['.py', '.pyw']).
        """
        pass

    @abstractmethod
    def parse_file_content(self, file_path: str, file_content: str) -> List[CodeEntity]:
        """
        Parses the content of a single file and extracts CodeEntity objects.

        Args:
            file_path: The path to the file being parsed.
            file_content: The actual text content of the file.

        Returns:
            A list of CodeEntity objects representing the structured elements found in the file.
        """
        pass

    def resolve_dependencies(
        self, 
        entities_in_file: List[CodeEntity], 
        all_entities_map: Dict[str, CodeEntity],
        # Potential future: accessor for entities not yet in all_entities_map
        # all_repo_entities_accessor: Callable[[str], Optional[CodeEntity]]
    ) -> None:
        """
        (Optional but recommended) Post-processing step to resolve dependencies between entities.
        This method can update the `dependencies` and `dependents` fields of the CodeEntity objects.
        It might be called after all files of a certain language (or all files in repo) are parsed.

        Args:
            entities_in_file: The list of entities parsed from the current file.
            all_entities_map: A dictionary view of all entities currently in the AbstractCodeKnowledgeBase,
                              mapping entity ID to CodeEntity. This allows looking up potential dependencies.
        """
        # Default implementation does nothing.
        # Parsers can override this to implement dependency resolution logic.
        pass


class PythonParser(ILanguageParser):
    """
    A parser for Python source code using the built-in `ast` module.
    """

    def get_supported_extensions(self) -> List[str]:
        return [".py", ".pyw"]

    def _generate_entity_id(self, filepath: str, qualified_name: str) -> str:
        """Generates a unique ID for an entity."""
        return f"{filepath}::{qualified_name}"

    def parse_file_content(self, file_path: str, file_content: str) -> List[CodeEntity]:
        """Parses Python file content into CodeEntity objects."""
        entities: List[CodeEntity] = []
        try:
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError as e:
            # print(f"Syntax error parsing {file_path}: {e}")
            # Create a simple file entity to represent the errored file
            return [
                CodeEntity(
                    id=self._generate_entity_id(file_path, file_path), # File itself as qualified name
                    entity_type="file",
                    name=file_path.split('/')[-1],
                    qualified_name=file_path,
                    language="python",
                    filepath=file_path,
                    start_line=1,
                    end_line=len(file_content.splitlines()),
                    docstring=f"Error parsing file: {e}",
                    metadata={"parsing_error": str(e)}
                )
            ]

        # Create a top-level file entity
        file_docstring = ast.get_docstring(tree, clean=True)
        file_entity_id = self._generate_entity_id(file_path, file_path) # File path as its own qualified name
        file_entity = CodeEntity(
            id=file_entity_id,
            entity_type="file",
            name=file_path.split('/')[-1],
            qualified_name=file_path,
            language="python",
            filepath=file_path,
            start_line=1, 
            end_line=tree.end_lineno if hasattr(tree, 'end_lineno') and tree.end_lineno is not None else len(file_content.splitlines()),
            docstring=file_docstring,
            raw_text_snippet=file_content # Or a snippet for very large files
        )
        entities.append(file_entity)

        # Visitor to extract entities
        visitor = PythonASTVisitor(file_path, file_entity_id)
        visitor.visit(tree)
        entities.extend(visitor.entities)
        
        # Populate children_ids for the file entity
        file_entity.children_ids = [e.id for e in visitor.entities if e.parent_id == file_entity_id]

        return entities

    def resolve_dependencies(
        self, 
        entities_in_file: List[CodeEntity], 
        all_entities_map: Dict[str, CodeEntity]
    ) -> None:
        """
        Resolves import dependencies for Python entities.
        This is a basic implementation focusing on module-level imports.
        More sophisticated resolution (e.g., specific function/class imports) can be added.
        """
        for entity in entities_in_file:
            if entity.entity_type == "import_statement" and "imported_module_names" in entity.metadata:
                for imported_module_name in entity.metadata["imported_module_names"]:
                    # Try to find a corresponding file or module entity in the KB
                    # This is a simplified search. A real system might need to resolve relative imports,
                    # search sys.path, or understand project structure.
                    
                    # Attempt 1: Direct match with a file entity (module file)
                    # Assuming module 'x.y.z' corresponds to file 'x/y/z.py' or 'x/y/z/__init__.py'
                    potential_module_path_py = imported_module_name.replace('.', '/') + ".py"
                    potential_module_path_init = imported_module_name.replace('.', '/') + "/__init__.py"
                    
                    found_dependency_id = None
                    for kb_entity_id, kb_entity in all_entities_map.items():
                        if kb_entity.entity_type == "file" and \
                           (kb_entity.filepath.endswith(potential_module_path_py) or \
                            kb_entity.filepath.endswith(potential_module_path_init)):
                            found_dependency_id = kb_entity_id
                            break
                    
                    if found_dependency_id and found_dependency_id not in entity.dependencies:
                        entity.dependencies.append(found_dependency_id)
                        # Also update the dependent's list
                        dependent_entity = all_entities_map.get(found_dependency_id)
                        if dependent_entity and entity.id not in dependent_entity.dependents:
                            dependent_entity.dependents.append(entity.id)


class PythonASTVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str, file_entity_id: str):
        self.filepath = filepath
        self.file_entity_id = file_entity_id # ID of the parent file entity
        self.entities: List[CodeEntity] = []
        self.current_parent_id: str = file_entity_id 
        self.current_namespace: str = "" # For qualified names

    def _generate_entity_id(self, qualified_name: str) -> str:
        return f"{self.filepath}::{qualified_name}"

    def _get_node_text(self, node: ast.AST, source_lines: List[str]) -> str:
        # ast.unparse (Python 3.9+) is better, but for broader compatibility:
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and \
           hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
            start_line = node.lineno -1
            end_line = node.end_lineno -1
            
            if start_line == end_line:
                return source_lines[start_line][node.col_offset:node.end_col_offset]
            else:
                lines = [source_lines[start_line][node.col_offset:]]
                for i in range(start_line + 1, end_line):
                    lines.append(source_lines[i])
                lines.append(source_lines[end_line][:node.end_col_offset])
                return "\n".join(lines)
        return "Error: Could not extract source text."


    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Determine if it's a method or a function based on current_parent_id
        parent_entity = next((e for e in self.entities if e.id == self.current_parent_id), None)
        is_method = parent_entity and parent_entity.entity_type == "class"
        entity_type = "method" if is_method else "function"
        
        qualified_name = f"{self.current_namespace}.{node.name}" if self.current_namespace else node.name
        entity_id = self._generate_entity_id(qualified_name)
        
        docstring = ast.get_docstring(node, clean=True)
        # raw_text = self._get_node_text(node, source_lines) # Requires source_lines

        params = [arg.arg for arg in node.args.args]
        # More detailed metadata can be added for parameters, return type annotations etc.
        metadata = {"parameters": params}
        if node.returns:
            # This captures the annotation, not the resolved type
            metadata["return_type_annotation"] = ast.dump(node.returns) if isinstance(node.returns, ast.AST) else str(node.returns)


        code_entity = CodeEntity(
            id=entity_id,
            entity_type=entity_type,
            name=node.name,
            qualified_name=qualified_name,
            language="python",
            filepath=self.filepath,
            start_line=node.lineno,
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno, # end_lineno can be None
            docstring=docstring,
            # raw_text_snippet=raw_text, # Add if source_lines is available
            parent_id=self.current_parent_id,
            metadata=metadata
        )
        self.entities.append(code_entity)
        
        # For nested functions/classes, update parent and namespace
        # This basic visitor doesn't handle deeply nested structures perfectly for namespace
        # but sets parent_id correctly.
        # A more robust approach would use a stack for current_parent_id and current_namespace.
        # For now, children of functions are not further processed by this simple visitor.
        # self.generic_visit(node) # To visit children like nested functions/classes
        # For simplicity, we are not recursing into function bodies for more entities here.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Similar to FunctionDef, but can add metadata for async
        self.visit_FunctionDef(node) # Reuse logic, then specialize
        # Find the entity just added and update its metadata
        for entity in reversed(self.entities):
            if entity.name == node.name and entity.start_line == node.lineno:
                entity.metadata["is_async"] = True
                break

    def visit_ClassDef(self, node: ast.ClassDef):
        qualified_name = f"{self.current_namespace}.{node.name}" if self.current_namespace else node.name
        entity_id = self._generate_entity_id(qualified_name)
        docstring = ast.get_docstring(node, clean=True)
        # raw_text = self._get_node_text(node, source_lines)

        base_names = [ast.dump(b) if isinstance(b, ast.AST) else str(b) for b in node.bases] # Simplified base class names

        class_entity = CodeEntity(
            id=entity_id,
            entity_type="class",
            name=node.name,
            qualified_name=qualified_name,
            language="python",
            filepath=self.filepath,
            start_line=node.lineno,
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
            docstring=docstring,
            # raw_text_snippet=raw_text,
            parent_id=self.current_parent_id,
            metadata={"bases": base_names}
        )
        self.entities.append(class_entity)

        # Process children of the class
        original_parent_id = self.current_parent_id
        original_namespace = self.current_namespace
        self.current_parent_id = entity_id
        self.current_namespace = qualified_name
        
        children_ids = []
        for child_node in node.body:
            # Temporarily store entities generated by visiting children
            # to correctly assign children_ids
            child_start_index = len(self.entities)
            self.visit(child_node)
            for i in range(child_start_index, len(self.entities)):
                # Ensure these new entities have the correct parent_id
                self.entities[i].parent_id = entity_id 
                children_ids.append(self.entities[i].id)
        
        class_entity.children_ids = children_ids

        self.current_parent_id = original_parent_id
        self.current_namespace = original_namespace

    def visit_Import(self, node: ast.Import):
        imported_module_names = []
        for alias in node.names:
            imported_module_names.append(alias.name)
        
        entity_id = self._generate_entity_id(f"import_{node.lineno}_{node.col_offset}")
        code_entity = CodeEntity(
            id=entity_id,
            entity_type="import_statement",
            name=", ".join(imported_module_names), # Simple name for the import
            qualified_name=f"{self.filepath}::import_{node.lineno}", # Unique enough
            language="python",
            filepath=self.filepath,
            start_line=node.lineno,
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
            parent_id=self.current_parent_id, # Import belongs to its containing scope (file or class/func)
            metadata={"imported_module_names": imported_module_names, "type": "direct"}
        )
        self.entities.append(code_entity)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module if node.module else "" # Relative imports might have module=None
        imported_names = []
        for alias in node.names:
            imported_names.append(alias.name)

        entity_id = self._generate_entity_id(f"import_from_{node.lineno}_{node.col_offset}")
        code_entity = CodeEntity(
            id=entity_id,
            entity_type="import_statement",
            name=f"from {module_name} import {', '.join(imported_names)}",
            qualified_name=f"{self.filepath}::import_from_{node.lineno}",
            language="python",
            filepath=self.filepath,
            start_line=node.lineno,
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
            parent_id=self.current_parent_id,
            metadata={
                "imported_from_module": module_name, 
                "imported_names": imported_names,
                "level": node.level, # For relative imports
                "type": "from"
            }
        )
        self.entities.append(code_entity)

    # Can add visit_Assign for global/class variables, etc.
    # For simplicity, this visitor focuses on functions, classes, and imports.

# Example of how to get source lines if needed by the visitor for raw_text_snippet
# This would typically be done in the `parse_file_content` method of the parser.
# source_lines = file_content.splitlines()
# visitor = PythonASTVisitor(file_path, file_entity_id, source_lines)
# visitor.visit(tree)
