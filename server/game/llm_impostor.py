"""LLM-powered impostor statement generator via HF Inference API."""
import os
from typing import Optional


def generate_llm_impostor_statement(
    impostor_name: str,
    fake_location: str,
    fake_task: str,
    kill_location: str,
    lie_type: str,
    scripted_statement: str,
    hf_token: Optional[str] = None,
) -> str:
    """
    Generate a natural impostor statement using an LLM.
    Falls back to scripted_statement if API unavailable.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return scripted_statement

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)

        lie_hints = {
            "location_clash": "Claim you were alone so no one can contradict you.",
            "task_impossibility": "Mention doing a specific task confidently.",
            "timing_contradiction": "Mention seeing another player somewhere.",
            "kill_proximity": "Emphasize you were far from the body location.",
            "corroboration_gap": "You were alone, no witnesses to contradict you.",
            "duo_cover": "Mention being with another player the whole time.",
        }
        hint = lie_hints.get(lie_type, "Sound natural and not suspicious.")

        prompt = f"""You are {impostor_name} in a game of Among Us. You are the IMPOSTOR.
A body was just found in {kill_location}. You need to give a convincing 1-2 sentence alibi.
Your alibi: you were in {fake_location.title()} doing {fake_task}.
Tip: {hint}
Respond ONLY with your alibi statement. Sound like a normal player, not guilty. No quotes."""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=80,
            temperature=0.8,
        )
        result = response.choices[0].message.content.strip().split("\n")[0].strip()
        if len(result) > 20:
            return result
        return scripted_statement

    except Exception:
        return scripted_statement
