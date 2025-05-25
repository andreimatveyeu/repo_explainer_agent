from typing import Dict, Any, Optional, List

from src.core.state import RepoExplainerState

def response_generator_node(state: RepoExplainerState) -> RepoExplainerState:
    """
    Formats the final response for the user.
    In Phase 1, this is a simple node that primarily passes through 
    `generated_explanation` or `overall_summary` if the intent was for an overview.
    It also incorporates error messages if present.
    
    The output of this node is not directly stored back into the main graph state's
    `generated_explanation` but is considered the "final output" to the user for a given turn.
    However, for LangGraph to work, it must return a dictionary that can update the state.
    We can add a specific key like `final_response_for_user` to the state if needed,
    or simply ensure `generated_explanation` holds the final formatted text.
    For now, let's assume it prepares `generated_explanation` as the final output.
    """
    print("--- Running Node: response_generator_node ---")
    
    error_message: Optional[str] = state.get("error_message")
    generated_explanation: Optional[str] = state.get("generated_explanation")
    overall_summary: Optional[str] = state.get("overall_summary")
    parsed_query_intent: Optional[str] = state.get("parsed_query_intent")
    
    final_response_text: str = ""
    updated_state: Dict[str, Any] = {}

    if error_message:
        print(f"Error message present: {error_message}")
        final_response_text = f"An error occurred: {error_message}"
    elif parsed_query_intent == "explain_repository_overview" and overall_summary:
        print("Intent is repository overview, using overall_summary.")
        final_response_text = overall_summary
    elif generated_explanation:
        print("Using generated_explanation for response.")
        final_response_text = generated_explanation
    elif overall_summary: # Fallback to overall_summary if no specific explanation but summary exists
        print("No specific explanation, falling back to overall_summary.")
        final_response_text = f"Here's a general overview of the repository:\n{overall_summary}"
    else:
        print("No specific explanation, error, or summary available.")
        final_response_text = "I'm sorry, I couldn't generate a specific response for your query at this time."

    # In a more complex agent, this node would also handle formatting visualizations,
    # adding suggestions for follow-up questions, etc.
    
    # For now, we'll update 'generated_explanation' to be this final response.
    # Or, we could introduce a new state key like 'final_user_output'.
    # Let's stick to updating 'generated_explanation' for simplicity in Phase 1,
    # and also log it to explanation_history.
    
    updated_state["generated_explanation"] = final_response_text
    
    # Update explanation_history
    current_query = state.get("user_query", "N/A")
    history_entry = {"query": current_query, "explanation": final_response_text}
    
    explanation_history: List[Dict[str,str]] = state.get("explanation_history", [])
    # Make a new list to ensure state update is detected if explanation_history was None
    new_explanation_history = list(explanation_history) 
    new_explanation_history.append(history_entry)
    updated_state["explanation_history"] = new_explanation_history

    print(f"Final response generated: {final_response_text[:200]}...")
    # This node effectively "ends" a turn by preparing the response.
    # The graph will then typically loop back to `user_query_parser` for the next input.
    
    return {**state, **updated_state}


if __name__ == '__main__':
    # Example Usage
    
    # Test case 1: With a generated explanation
    state_with_explanation: RepoExplainerState = {
        "user_query": "Explain file X.",
        "generated_explanation": "File X contains important logic for feature Y.",
        "explanation_history": []
    }
    print("\n--- Test 1: With generated explanation ---")
    result1 = response_generator_node(state_with_explanation)
    print(f"Response: {result1.get('generated_explanation')}")
    assert result1.get('generated_explanation') == "File X contains important logic for feature Y."
    assert len(result1.get('explanation_history', [])) == 1
    assert result1['explanation_history'][0]['explanation'] == "File X contains important logic for feature Y."

    # Test case 2: With an error message
    state_with_error: RepoExplainerState = {
        "user_query": "Explain file Y.",
        "error_message": "File Y not found.",
        "explanation_history": []
    }
    print("\n--- Test 2: With error message ---")
    result2 = response_generator_node(state_with_error)
    print(f"Response: {result2.get('generated_explanation')}")
    assert "An error occurred: File Y not found." in result2.get('generated_explanation', '')
    assert len(result2.get('explanation_history', [])) == 1

    # Test case 3: Intent is overview, overall_summary is present
    state_for_overview: RepoExplainerState = {
        "user_query": "Tell me about the repo.",
        "parsed_query_intent": "explain_repository_overview",
        "overall_summary": "This is a great project about cats.",
        "generated_explanation": None, # No specific file explanation
        "explanation_history": []
    }
    print("\n--- Test 3: Intent is overview ---")
    result3 = response_generator_node(state_for_overview)
    print(f"Response: {result3.get('generated_explanation')}")
    assert result3.get('generated_explanation') == "This is a great project about cats."
    assert len(result3.get('explanation_history', [])) == 1

    # Test case 4: No specific explanation, but overall_summary exists (fallback)
    state_fallback_summary: RepoExplainerState = {
        "user_query": "What is foo?", # Assume this didn't lead to a specific generated_explanation
        "parsed_query_intent": "find_code_entity", # Not an overview intent
        "overall_summary": "This project is about foo and bar.",
        "generated_explanation": None,
        "explanation_history": []
    }
    print("\n--- Test 4: Fallback to overall_summary ---")
    result4 = response_generator_node(state_fallback_summary)
    print(f"Response: {result4.get('generated_explanation')}")
    assert "Here's a general overview of the repository:\nThis project is about foo and bar." in result4.get('generated_explanation', '')
    assert len(result4.get('explanation_history', [])) == 1
    
    # Test case 5: No relevant information
    state_no_info: RepoExplainerState = {
        "user_query": "...",
        "explanation_history": [{"query": "prev q", "explanation": "prev ans"}]
    }
    print("\n--- Test 5: No relevant info ---")
    result5 = response_generator_node(state_no_info)
    print(f"Response: {result5.get('generated_explanation')}")
    assert "I'm sorry, I couldn't generate a specific response" in result5.get('generated_explanation', '')
    assert len(result5.get('explanation_history', [])) == 2 # Appends to existing history

    print("\nresponse_generator_node tests completed.")
