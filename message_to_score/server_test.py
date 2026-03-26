import json
import sys
from urllib import error, request


SERVER_URL = "http://127.0.0.1:8000/evaluate"


def main() -> int:
    payload = {
        "scene": 138,
        "messages": [
            {"role": "user", "content": "我总是分不清渗透和扩散。"},
            {
                "role": "assistant",
                "content": "可以从定义、发生条件和生活例子三个角度来区分渗透和扩散。",
            },
        ],
    }

    req = request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}")
        return 1
    except error.URLError as exc:
        print(f"Request failed: {exc}")
        print("Make sure `python server.py` is running first.")
        return 1

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        print(body)
        return 1

    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
