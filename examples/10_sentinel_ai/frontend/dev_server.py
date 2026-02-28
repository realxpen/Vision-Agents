import json
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from getstream import Stream


FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parents[2]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/frontend-config":
            self._handle_frontend_config(parsed.query)
            return
        super().do_GET()

    def _handle_frontend_config(self, query: str) -> None:
        params = parse_qs(query)
        default_user_id = os.getenv("SENTINEL_FRONTEND_USER_ID", "user-demo-agent")
        default_call_type = os.getenv("SENTINEL_FRONTEND_CALL_TYPE", "default")
        default_call_id = os.getenv("SENTINEL_FRONTEND_CALL_ID", "sentinel-live")
        user_id = params.get("user_id", [default_user_id])[0]
        if user_id == "user-demo-agent":
            user_id = f"user-demo-agent-{int(time.time())}"
        call_type = params.get("call_type", [default_call_type])[0]
        call_id = params.get("call_id", [default_call_id])[0]

        api_key = os.getenv("STREAM_API_KEY")
        api_secret = os.getenv("STREAM_API_SECRET")
        if not api_key or not api_secret:
            self._json(
                500,
                {
                    "error": "Missing STREAM_API_KEY or STREAM_API_SECRET in .env",
                },
            )
            return

        client = Stream(api_key=api_key, api_secret=api_secret)
        token = client.create_token(user_id, expiration=3600)
        self._json(
            200,
            {
                "apiKey": api_key,
                "userId": user_id,
                "token": token,
                "callType": call_type,
                "callId": call_id,
            },
        )

    def _json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    host = os.getenv("SENTINEL_FRONTEND_HOST", "0.0.0.0")
    port = int(os.getenv("SENTINEL_FRONTEND_PORT", "5500"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Sentinel frontend dev server: http://{host}:{port}/preview.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
