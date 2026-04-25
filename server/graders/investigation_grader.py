from typing import Dict, Any, List


def grade_investigation(
    tool_calls: List[Dict[str, Any]],
    relevant_tools: List[str],
    scenario
) -> Dict[str, Any]:
    score = 0.0
    flags = {}

    seen_calls = set()
    redundant_count = 0
    read_statements_used = False
    cross_reference_used = False

    for call in tool_calls:
        tool_name = call.get("tool_name", "")
        tool_args = call.get("tool_args", {})
        call_key = f"{tool_name}:{str(sorted(tool_args.items()) if tool_args else [])}"

        if call_key in seen_calls:
            redundant_count += 1
            score -= 0.03
            continue
        seen_calls.add(call_key)

        if tool_name == "read_statements":
            if not read_statements_used:
                score += 0.03
                read_statements_used = True

        elif tool_name == "check_player_presence":
            location = tool_args.get("location", "").upper()
            relevant = any(location in rt for rt in relevant_tools)
            if relevant:
                score += 0.05
                flags[f"relevant_presence_check_{location}"] = True
            else:
                score += 0.01

        elif tool_name == "check_location_tasks":
            location = tool_args.get("location", "").upper()
            relevant = any(location in rt for rt in relevant_tools)
            if relevant:
                score += 0.03
                flags[f"relevant_task_check_{location}"] = True

        elif tool_name == "cross_reference":
            if not cross_reference_used:
                score += 0.05
                cross_reference_used = True

        elif tool_name == "check_adjacency":
            loc1 = tool_args.get("location1", "").upper()
            loc2 = tool_args.get("location2", "").upper()
            relevant = any((loc1 in rt or loc2 in rt) for rt in relevant_tools)
            if relevant:
                score += 0.03

    if redundant_count > 0:
        flags["redundant_calls"] = redundant_count

    score = max(-0.2, min(0.5, score))
    return {"investigation_score": score, "flags": flags, "tool_call_count": len(tool_calls)}
