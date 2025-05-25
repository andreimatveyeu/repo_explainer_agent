import os
import json
from typing import Dict, List, Any, Optional
import google.generativeai as genai

# Configure the generative AI model
# The API key is expected to be set as an environment variable GOOGLE_AISTUDIO_API_KEY
genai.configure(api_key=os.environ.get("GOOGLE_AISTUDIO_API_KEY"))

def call_llm_for_summary(
    prompt: str,
    context_chunks: Optional[List[str]] = None,
    max_tokens: int = 500,
    temperature: float = 0.3,
    model_name: str = "gemini-2.0-flash-001"
) -> Optional[str]:
    """
    Calls the Google Gemini LLM for summarization or explanation.

    Args:
        prompt: The main prompt for the LLM.
        context_chunks: Optional list of text chunks to provide as context.
        max_tokens: Maximum tokens for the LLM response.
        temperature: Temperature for LLM generation.
        model_name: The specific LLM model to use.

    Returns:
        The LLM's generated text response, or None if an error occurs.
    """
    full_prompt = prompt
    if context_chunks:
        full_prompt += "\n\nRelevant Context:\n" + "\n---\n".join(context_chunks)

    try:
        model = genai.GenerativeModel(model_name)
        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        # print(f"--- LLM Call (call_llm_for_summary) ---")
        # print(f"Model: {model_name}")
        # print(f"Prompt (first 100 chars): {full_prompt[:100]}...")
        # print(f"GenerationConfig: {config}")

        response = model.generate_content(
            full_prompt,
            generation_config=config,
        )
        
        # Check finish reason and parts before accessing response.text
        if response.candidates and \
           response.candidates[0].content and \
           response.candidates[0].content.parts:
            
            finish_reason = response.candidates[0].finish_reason
            # Compare with the STOP member of the actual finish_reason enum type
            if finish_reason != type(finish_reason).STOP:
                 print(f"LLM generation for summary finished with reason: {finish_reason.name} ({finish_reason.value}). Response might be incomplete or missing.")
            
            # It's generally safer to access parts directly if needed,
            # but response.text should work if parts[0] is text.
            # If finish_reason was not STOP, .text might still be empty or partial.
            return response.text 
        else:
            finish_reason_name = "UNKNOWN"
            finish_reason_value = -1
            # prompt_feedback_info = "N/A" # Removed
            # candidate_safety_ratings_info = "N/A" # Removed

            if response.candidates and response.candidates[0].finish_reason:
                finish_reason_name = response.candidates[0].finish_reason.name
                finish_reason_value = response.candidates[0].finish_reason.value
            
            # if hasattr(response, 'prompt_feedback') and response.prompt_feedback: # Removed
            #     prompt_feedback_info = f"Block Reason: {response.prompt_feedback.block_reason}, " \
            #                            f"Safety Ratings: {response.prompt_feedback.safety_ratings}" # Removed
            
            # if response.candidates and hasattr(response.candidates[0], 'safety_ratings') and response.candidates[0].safety_ratings: # Removed
            #     candidate_safety_ratings_info = str(response.candidates[0].safety_ratings) # Removed

            print(f"Error calling LLM for summary: No valid content parts in response. Finish reason: {finish_reason_name} ({finish_reason_value})")
            # print(f"Prompt Feedback (summary): {prompt_feedback_info}") # Removed
            # print(f"Candidate Safety Ratings (summary): {candidate_safety_ratings_info}") # Removed
            # print(f"Full LLM response object for debugging (summary): {response}") # Re-commented
            return None
            
    except Exception as e:
        # This catches other errors, e.g., API connection issues, or if response.text itself errors
        # despite the checks (though less likely with the checks).
        print(f"Error calling LLM for summary: {e}")
        # print(f"Full LLM response object at time of exception: {response if 'response' in locals() else 'Response object not available'}")
        return None

def call_llm_for_structured_output(
    prompt: str,
    output_format_description: str,
    context_chunks: Optional[List[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.1,
    model_name: str = "gemini-2.0-flash-001"
) -> Optional[Dict[str, Any]]:
    """
    Calls the Google Gemini LLM for structured data (e.g., JSON) output.
    """
    full_prompt = prompt
    if context_chunks:
        full_prompt += "\n\nRelevant Context:\n" + "\n---\n".join(context_chunks)
    full_prompt += f"\n\nOutput Format Instructions:\n{output_format_description}"
    
    response_text_for_json_parsing = "" # Initialize for logging in case of early exit

    try:
        model = genai.GenerativeModel(model_name)
        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            # Consider adding response_mime_type="application/json" if the model/API supports it
            # for more reliable JSON output, though this might require specific model versions.
        )
        # print(f"--- LLM Call (call_llm_for_structured_output) ---")
        # print(f"Model: {model_name}")
        # print(f"Prompt (first 100 chars): {full_prompt[:100]}...")
        # print(f"GenerationConfig: {config}")
        
        response = model.generate_content(
            full_prompt,
            generation_config=config,
        )

        if response.candidates and \
           response.candidates[0].content and \
           response.candidates[0].content.parts:
            
            finish_reason = response.candidates[0].finish_reason
            # Compare with the STOP member of the actual finish_reason enum type
            if finish_reason != type(finish_reason).STOP:
                 print(f"LLM generation for structured output finished with reason: {finish_reason.name} ({finish_reason.value}). Response might be incomplete or malformed.")

            response_text_for_json_parsing = response.text # Get text from the first part
            
            # Attempt to parse JSON
            return json.loads(response_text_for_json_parsing)
        else:
            finish_reason_name = "UNKNOWN"
            finish_reason_value = -1
            # prompt_feedback_info = "N/A" # Removed
            # candidate_safety_ratings_info = "N/A" # Removed

            if response.candidates and response.candidates[0].finish_reason:
                finish_reason_name = response.candidates[0].finish_reason.name
                finish_reason_value = response.candidates[0].finish_reason.value

            # if hasattr(response, 'prompt_feedback') and response.prompt_feedback: # Removed
            #     prompt_feedback_info = f"Block Reason: {response.prompt_feedback.block_reason}, " \
            #                            f"Safety Ratings: {response.prompt_feedback.safety_ratings}" # Removed

            # if response.candidates and hasattr(response.candidates[0], 'safety_ratings') and response.candidates[0].safety_ratings: # Removed
            #     candidate_safety_ratings_info = str(response.candidates[0].safety_ratings) # Removed

            print(f"Error calling LLM for structured output: No valid content parts in response. Finish reason: {finish_reason_name} ({finish_reason_value})")
            # print(f"Prompt Feedback (structured): {prompt_feedback_info}") # Removed
            # print(f"Candidate Safety Ratings (structured): {candidate_safety_ratings_info}") # Removed
            # print(f"Full LLM response object for debugging (structured): {response}") # Re-commented
            return None
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM response text that failed to parse: '{response_text_for_json_parsing}'")
        return None
    except Exception as e: # Catches other errors like API issues or unexpected response structure
        print(f"Error calling LLM for structured output: {e}")
        # print(f"Full LLM response object at time of exception: {response if 'response' in locals() else 'Response object not available'}")
        return None
