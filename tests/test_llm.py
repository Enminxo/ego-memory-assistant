import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from egomem.describer import VLMDescriber
from egomem.llm import ChatClient


def serve(reply: str):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": reply}}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_describer_sends_frames_and_disables_thinking():
    server, seen = serve("I open the fridge. I take out the")
    try:
        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "some-vlm")
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
        result = VLMDescriber(client).describe(frames, 15, 30)
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "some-vlm" and body["reasoning_effort"] == "none"
    content = body["messages"][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 3
    assert "00:15-00:30" in content[0]["text"]
    assert result.text == "I open the fridge."  # trailing fragment trimmed
    assert result.prompt_tokens == 10
