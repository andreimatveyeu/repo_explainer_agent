from typing import List, Dict, Any, Optional, Literal, Callable
from pydantic import BaseModel, Field

class CodeEntity(BaseModel):
    """
    Represents a structural element within a codebase, such as a file, class, function, etc.
    This model is designed to be language-agnostic.
    """
    id: str = Field(..., description="Unique identifier for the code entity. Strategy: 'filepath::qualified_name' or hash for elements without a clear qualified name.")
    entity_type: Literal[
        "file", "module", "namespace", "class", "interface", "struct", 
        "enum", "function", "method", "constructor", "variable", "constant", 
        "property", "import_statement", "comment_block", "decorator", "type_alias"
    ] = Field(..., description="The type of the code entity.")
    name: str = Field(..., description="The simple name of the entity (e.g., 'MyClass', 'my_function').")
    qualified_name: Optional[str] = Field(None, description="The fully qualified name of the entity (e.g., 'my_module.MyClass.my_method'). Path-like for files.")
    language: str = Field(..., description="The programming language of this entity (e.g., 'python', 'javascript').")
    filepath: str = Field(..., description="The absolute or project-relative path to the file containing this entity.")
    start_line: int = Field(..., description="The 1-indexed starting line number of the entity in the source file.")
    end_line: int = Field(..., description="The 1-indexed ending line number of the entity in the source file.")
    
    raw_text_snippet: Optional[str] = Field(None, description="The raw source code text of the entity. May be omitted for brevity for large entities.")
    summary: Optional[str] = Field(None, description="An LLM-generated or parser-extracted summary of the entity's purpose and functionality.")
    docstring: Optional[str] = Field(None, description="The verbatim docstring or header comment associated with the entity, if any.")
    
    dependencies: List[str] = Field(default_factory=list, description="List of IDs of other CodeEntity objects that this entity directly depends on (e.g., calls, imports, inherits from).")
    dependents: List[str] = Field(default_factory=list, description="List of IDs of other CodeEntity objects that directly depend on this entity.")
    
    parent_id: Optional[str] = Field(None, description="ID of the parent CodeEntity (e.g., a class containing a method, a file containing a function).")
    children_ids: List[str] = Field(default_factory=list, description="List of IDs of CodeEntity objects that are children of this entity (e.g., methods within a class, functions within a file).")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Language-specific or type-specific metadata. E.g., for a function: {'parameters': [...], 'return_type': 'str'}, for a class: {'is_abstract': False}.")

    class Config:
        use_enum_values = True


class AbstractCodeKnowledgeBase(BaseModel):
    """
    A structured, queryable representation of the codebase, composed of CodeEntity objects.
    This knowledge base is designed to be language-agnostic at its query interface.
    """
    entities: Dict[str, CodeEntity] = Field(default_factory=dict, description="A dictionary mapping entity IDs to CodeEntity objects.")
    file_paths: List[str] = Field(default_factory=list, description="A list of all file paths processed in the repository.")
    detected_languages: List[str] = Field(default_factory=list, description="List of programming languages detected in the repository.")

    def add_entity(self, entity: CodeEntity) -> None:
        """Adds a CodeEntity to the knowledge base."""
        if entity.id in self.entities:
            # Potentially update or merge, for now, let's overwrite with a warning or raise error
            # print(f"Warning: Entity with ID {entity.id} already exists. Overwriting.")
            pass
        self.entities[entity.id] = entity
        if entity.filepath not in self.file_paths:
            self.file_paths.append(entity.filepath)
        if entity.language not in self.detected_languages:
            self.detected_languages.append(entity.language)

    def get_entity(self, entity_id: str) -> Optional[CodeEntity]:
        """Retrieves a CodeEntity by its ID."""
        return self.entities.get(entity_id)

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
        filepath: Optional[str] = None,
        custom_filter: Optional[Callable[[CodeEntity], bool]] = None
    ) -> List[CodeEntity]:
        """
        Finds entities based on specified criteria.
        All criteria are ANDed together.
        """
        results = []
        for entity in self.entities.values():
            match = True
            if name is not None and entity.name != name and entity.qualified_name != name:
                match = False
            if entity_type is not None and entity.entity_type != entity_type:
                match = False
            if language is not None and entity.language != language:
                match = False
            if filepath is not None and entity.filepath != filepath:
                match = False
            if custom_filter is not None and not custom_filter(entity):
                match = False
            
            if match:
                results.append(entity)
        return results

    def get_dependencies(self, entity_id: str) -> List[CodeEntity]:
        """Retrieves all direct dependencies of a given entity."""
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        return [self.get_entity(dep_id) for dep_id in entity.dependencies if self.get_entity(dep_id)]

    def get_dependents(self, entity_id: str) -> List[CodeEntity]:
        """Retrieves all direct dependents of a given entity."""
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        # This requires dependents to be populated. Alternatively, search all entities.
        # For now, assuming dependents field is populated correctly by parsers/resolvers.
        return [self.get_entity(dep_id) for dep_id in entity.dependents if self.get_entity(dep_id)]

    def get_file_entities(self, filepath: str) -> List[CodeEntity]:
        """Retrieves all CodeEntity objects belonging to a specific file."""
        return [entity for entity in self.entities.values() if entity.filepath == filepath]

    def get_entities_by_type(self, entity_type: str, language: Optional[str] = None) -> List[CodeEntity]:
        """Retrieves all entities of a specific type, optionally filtered by language."""
        return self.find_entities(entity_type=entity_type, language=language)

    def get_children(self, parent_id: str) -> List[CodeEntity]:
        """Retrieves all direct children of a given parent entity."""
        parent_entity = self.get_entity(parent_id)
        if not parent_entity:
            return []
        return [self.get_entity(child_id) for child_id in parent_entity.children_ids if self.get_entity(child_id)]

    def get_parent(self, child_id: str) -> Optional[CodeEntity]:
        """Retrieves the parent of a given child entity."""
        child_entity = self.get_entity(child_id)
        if not child_entity or not child_entity.parent_id:
            return None
        return self.get_entity(child_entity.parent_id)

    def update_dependencies_for_entity(self, entity_id: str, dependency_ids: List[str]):
        """Updates the dependencies for a specific entity and attempts to update dependents."""
        entity = self.get_entity(entity_id)
        if not entity:
            # print(f"Error: Entity {entity_id} not found for updating dependencies.")
            return

        # Remove old dependent links
        for old_dep_id in entity.dependencies:
            old_dependent_entity = self.get_entity(old_dep_id)
            if old_dependent_entity and entity_id in old_dependent_entity.dependents:
                old_dependent_entity.dependents.remove(entity_id)
        
        entity.dependencies = dependency_ids

        # Add new dependent links
        for dep_id in dependency_ids:
            dependent_entity = self.get_entity(dep_id)
            if dependent_entity and entity_id not in dependent_entity.dependents:
                dependent_entity.dependents.append(entity_id)

    def __str__(self) -> str:
        return (f"AbstractCodeKnowledgeBase(entities={len(self.entities)}, "
                f"files={len(self.file_paths)}, languages={self.detected_languages})")
