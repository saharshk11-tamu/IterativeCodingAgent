import os
import re
import signal
import sys
import time
import unittest
import importlib
import subprocess
from pathlib import Path
from http.client import HTTPConnection


SERVER_SCRIPT = Path(__file__).with_name("server.py")
HOST = "localhost"
PORT = 8000
BASE_URL = f"{HOST}:{PORT}"
STARTUP_TIMEOUT = 5  # seconds to wait for the server to become ready


class ServerProcess:
    """
    Helper context manager to start `python server.py` in a subprocess,
    wait until it's ready to accept connections, and ensure clean teardown.
    """
    def __init__(self):
        self.proc = None

    def __enter__(self):
        if not SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"{SERVER_SCRIPT} not found in the current directory")

        # Start server with unbuffered stdout/stderr to capture logs if needed
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(SERVER_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the server to start listening
        start_time = time.time()
        while time.time() - start_time < STARTUP_TIMEOUT:
            try:
                conn = HTTPConnection(HOST, PORT, timeout=0.2)
                conn.request("GET", "/")
                conn.getresponse()
                conn.close()
                break  # Successfully connected
            except OSError:
                time.sleep(0.1)
        else:
            # Server did not start within timeout
            self._dump_subprocess_output()
            self.proc.terminate()
            raise RuntimeError("Server did not start within expected time frame")

        return self.proc

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Attempt graceful shutdown via SIGINT / CTRL_BREAK_EVENT
        if self.proc.poll() is None:  # still running
            try:
                if os.name == "nt":
                    # On Windows, CTRL_BREAK_EVENT is used to mimic Ctrl+C in subprocesses
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=2)
            except Exception:
                # Force kill if graceful shutdown failed
                self.proc.kill()
        # Drain remaining output to avoid pipe deadlocks
        self._dump_subprocess_output()

    def _dump_subprocess_output(self):
        try:
            stdout, stderr = self.proc.communicate(timeout=0.5)
            if stdout:
                sys.stdout.write(stdout.decode(errors="replace"))
            if stderr:
                sys.stderr.write(stderr.decode(errors="replace"))
        except Exception:
            pass


class TestServerResponses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ServerProcess()
        cls.proc = cls.server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def _make_request(self, path="/"):
        conn = HTTPConnection(HOST, PORT, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()
        return response.status, body

    def test_root_returns_hello_world(self):
        """
        Requirement: The server responds to GET / with status 200
        and body exactly 'Hello, world!'.
        """
        status, body = self._make_request("/")
        self.assertEqual(status, 200, "GET / did not return HTTP 200")
        self.assertEqual(body, "Hello, world!", "GET / did not return expected body")

    def test_non_root_returns_404(self):
        """
        Requirement: The server responds to any non-root path with status 404.
        """
        for path in ["/unknown", "/foo", "/bar/baz"]:
            status, _ = self._make_request(path)
            self.assertEqual(status, 404, f"GET {path} did not return HTTP 404")


class TestServerShutdown(unittest.TestCase):
    def test_graceful_shutdown_on_keyboard_interrupt(self):
        """
        Requirement: The server shuts down cleanly within 2 seconds when a KeyboardInterrupt is received.
        """
        with ServerProcess() as proc:
            start_time = time.time()
            # Send Ctrl+C equivalent
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail("Server did not shut down within 2 seconds after KeyboardInterrupt")

            elapsed = time.time() - start_time
            self.assertLessEqual(elapsed, 2.0, "Shutdown took longer than 2 seconds")
            self.assertIsNotNone(proc.returncode, "Process did not terminate")
            # Ensure no uncaught exceptions (non-zero exit code indicates problem)
            self.assertEqual(
                proc.returncode,
                0,
                f"Server exited with non-zero status {proc.returncode}"
            )


class TestStandardLibraryImports(unittest.TestCase):
    def test_only_standard_library_imports_used(self):
        """
        Constraint: Only Python standard-library modules may be imported.
        This test parses import statements in server.py and ensures none
        reside in site-packages/dist-packages.
        """
        import_lines = re.findall(r'^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)', SERVER_SCRIPT.read_text(), re.MULTILINE)
        # Get unique top-level packages/modules
        top_level_modules = {name.split('.')[0] for name in import_lines}

        # Always allowed: 'server' may import from __future__ for compatibility
        top_level_modules.discard('__future__')

        site_paths = [Path(p).resolve() for p in sys.path if 'site-packages' in p or 'dist-packages' in p]

        offending_modules = []
        for mod_name in top_level_modules:
            try:
                mod = importlib.import_module(mod_name)
                mod_file = getattr(mod, '__file__', None)
                if mod_file is not None:
                    mod_path = Path(mod_file).resolve()
                    if any(str(mod_path).startswith(str(site_p)) for site_p in site_paths):
                        offending_modules.append(mod_name)
            except Exception:
                # If import fails, consider it non-standard or missing; mark as offending
                offending_modules.append(mod_name)

        self.assertEqual(
            offending_modules,
            [],
            f"Non-standard library modules imported: {offending_modules}"
        )


if __name__ == "__main__":
    unittest.main()