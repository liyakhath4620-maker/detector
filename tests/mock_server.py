import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MockOmniSightHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler that captures and validates POSTed PotholeRecord JSON payloads.
    """

    def log_message(self, format, *args):
        # Silence default stderr logging during automated tests
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Empty request body"}')
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Malformed JSON: {e}"}).encode("utf-8"))
            return

        # Validate required PotholeRecord schema fields
        required_fields = [
            "pothole_id",
            "timestamp",
            "confidence",
            "gps_fix_quality",
            "depth_quality_flag",
            "bounding_box",
        ]
        missing = [f for f in required_fields if f not in payload]
        if missing:
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Schema validation failed. Missing: {missing}"}).encode("utf-8"))
            return

        bbox = payload.get("bounding_box", {})
        if not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "bounding_box must contain x1, y1, x2, y2"}')
            return

        # Payload is valid: capture in server list
        self.server.received_records.append(payload)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "received", "success": true}')


class MockOmniSightServer:
    """
    Lightweight background HTTP receiver for testing edge detection pipeline.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        """
        Initializes the mock server.

        :param host: Host address (default 127.0.0.1).
        :param port: Port number (0 allocates an available ephemeral port).
        """
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.received_records: List[Dict[str, Any]] = []

    def start(self):
        """Starts the HTTP server on a background daemon thread."""
        if self.server is not None:
            return

        self.server = HTTPServer((self.host, self.port), MockOmniSightHandler)
        self.server.received_records = self.received_records
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"MockOmniSightServer running at {self.url}")

    def stop(self):
        """Stops the HTTP server and releases the socket."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("MockOmniSightServer stopped")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/api/potholes"

    def get_received_records(self) -> List[Dict[str, Any]]:
        """Returns a copy of all captured valid records."""
        return list(self.received_records)

    def clear(self):
        """Clears captured records."""
        self.received_records.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run_mock_server(host: str = "127.0.0.1", port: int = 8000):
    """Entry point to run standalone mock server for manual testing."""
    server = MockOmniSightServer(host, port)
    server.start()
    print(f"Mock server listening on {server.url}. Press Ctrl+C to stop.")
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        server.stop()
        print("\nMock server stopped.")


if __name__ == "__main__":
    run_mock_server()
