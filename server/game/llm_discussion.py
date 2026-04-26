"""LLM-powered discussion statement generator for crew and impostor players."""
import os
import random
from typing import Optional, List, Dict


def generate_discussion_statement(
    player_name: str,
    role: str,
    personality: str,
    all_statements: Dict[str, str],
    discussion_history: List[Dict],
    round_num: int,
    kill_location: str,
    fake_location: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    fallback = _scripted_discussion(
        player_name, role, all_statements, discussion_history, round_num, kill_location, fake_location
    )

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return fallback

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)

        statements_text = "\n".join(
            f"  {name}: \"{stmt}\"" for name, stmt in all_statements.items() if name != player_name
        )
        history_text = ""
        if discussion_history:
            history_text = "\n".join(
                f"  {d['player_name']}: \"{d['statement']}\"" for d in discussion_history[-6:]
            )

        if role == "impostor":
            accused = any(
                player_name.lower() in d.get("statement", "").lower()
                for d in discussion_history
            )
            loc = fake_location.title() if fake_location else "another room"
            if accused:
                prompt = (
                    f"You are {player_name} in Among Us. You are the IMPOSTOR but must not reveal it.\n"
                    f"A body was found in {kill_location}. You claimed you were in {loc}.\n"
                    f"Other players have accused you. Defend yourself in 1-2 sentences. "
                    f"Sound innocent, slightly frustrated, and deflect to someone else.\n"
                    f"What others said:\n{statements_text}\n"
                    f"Discussion so far:\n{history_text}\n"
                    f"Your defense (no quotes, 1-2 sentences only):"
                )
            else:
                others = [n for n in all_statements if n != player_name]
                target = random.choice(others) if others else "someone"
                prompt = (
                    f"You are {player_name} in Among Us. You are the IMPOSTOR but must not reveal it.\n"
                    f"A body was found in {kill_location}. You claimed you were in {loc}.\n"
                    f"Subtly point suspicion at {target} in 1-2 sentences. Sound natural and helpful.\n"
                    f"What others said:\n{statements_text}\n"
                    f"Discussion so far:\n{history_text}\n"
                    f"Your statement (no quotes, 1-2 sentences only):"
                )
        else:
            prompt = (
                f"You are {player_name} in Among Us. You are a CREWMATE trying to find the impostor.\n"
                f"A body was found in {kill_location}. Analyze statements and name who you find suspicious.\n"
                f"Be specific. 1-2 sentences only.\n"
                f"What others said:\n{statements_text}\n"
                f"Discussion so far:\n{history_text}\n"
                f"Your statement (no quotes, 1-2 sentences, name a specific player you suspect):"
            )

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=80,
            temperature=0.85,
        )
        result = response.choices[0].message.content.strip().split("\n")[0].strip()
        if len(result) > 15:
            return result
        return fallback
    except Exception:
        return fallback


def _scripted_discussion(
    player_name: str,
    role: str,
    all_statements: Dict[str, str],
    discussion_history: List[Dict],
    round_num: int,
    kill_location: str,
    fake_location: Optional[str] = None,
) -> str:
    others = [n for n in all_statements if n != player_name]

    if role == "impostor":
        accused = any(
            player_name.lower() in d.get("statement", "").lower()
            for d in discussion_history
        )
        if accused:
            loc = fake_location.title() if fake_location else "my room"
            candidates = [n for n in others]
            deflect = random.choice(candidates) if candidates else "someone else"
            return (
                f"I was in {loc} the whole time — stop wasting time on me. "
                f"Has anyone actually looked at what {deflect} said?"
            )
        else:
            target = random.choice(others) if others else "someone"
            return (
                f"I wasn't anywhere near {kill_location}. "
                f"Honestly, {target}'s alibi sounds off to me — can they explain that again?"
            )
    else:
        if round_num == 0:
            suspicious = None
            for name, stmt in all_statements.items():
                if name != player_name and kill_location.lower() in stmt.lower():
                    suspicious = name
                    break
            if not suspicious and others:
                suspicious = random.choice(others)
            return (
                f"Looking at everyone's statements, {suspicious}'s alibi doesn't add up to me. "
                f"We should press them harder before we vote."
            )
        else:
            counts: Dict[str, int] = {}
            for d in discussion_history:
                stmt_lower = d.get("statement", "").lower()
                for name in others:
                    if name.lower() in stmt_lower:
                        counts[name] = counts.get(name, 0) + 1
            if counts:
                top = max(counts, key=counts.get)
                return f"{top} keeps coming up — I'm locking in my vote on them."
            elif others:
                return f"Based on everything said, {random.choice(others)} is my best guess. Let's vote."
            else:
                return "We've talked enough — let's vote and trust our instincts."
