import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "v2-generation-check.json"

PROMPT = (
    "Product: Earbuds\n"
    "Battery: 30 hours\n"
    "Question: What battery life is listed?\n"
    "Answer:"
)

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "prompt": PROMPT,
    "expected_fact": "The listed battery life is 30 hours.",
    "manual_quality_review": "Pending",
}

exit_code = 1
start = time.perf_counter()

try:
    response = requests.post(
        "http://127.0.0.1:8002/generate",
        json={
            "prompt": PROMPT,
            "max_tokens": 80,
        },
        timeout=(5, 120),
    )

    report["status_code"] = response.status_code
    report["elapsed_seconds"] = round(
        time.perf_counter() - start,
        3,
    )

    try:
        body = response.json()
    except ValueError:
        body = {"raw_response": response.text[:2000]}

    report["server_response"] = body

    response.raise_for_status()

    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")

    answer = body.get("response")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The model returned an empty or invalid answer.")

    checks = {
        "correct_model": (
            body.get("model") == "BuyWisely-TinyGPT-v2"
        ),
        "input_not_truncated": (
            body.get("input_truncated") is False
        ),
        "nonempty_completion": bool(answer.strip()),
    }

    report["checks"] = checks
    report["structural_checks_passed"] = all(checks.values())

    print("HTTP status:", response.status_code)
    print("Checks:", checks)
    print("Elapsed seconds:", report["elapsed_seconds"])
    print("\nGenerated answer:")
    print(answer)

    print(
        "\nManual review: does the answer correctly state "
        "'30 hours' without inventing additional specifications?"
    )

    exit_code = 0 if all(checks.values()) else 1

except (requests.exceptions.RequestException, ValueError) as exc:
    report["error"] = str(exc)
    print("Check failed:", exc)

finally:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nReport saved:", REPORT_PATH)

raise SystemExit(exit_code)