import subprocess
import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Path to the built executable (adjust if needed)
EXE_PATH = Path("dist/ct2-translate-service.exe")

# Test language pairs (should match AVAILABLE_MODELS)
TEST_PAIRS = [
    ("en", "tr", "Hello world!"),
    ("tr", "en", "Merhaba dünya!"),
    ("en", "es", "How are you?"),
    ("es", "en", "¿Cómo estás?"),
]

TIMEOUT = 60  # seconds for model download/translation

def send_cmd(proc, cmd):
    proc.stdin.write((json.dumps(cmd) + "\n").encode("utf-8"))
    proc.stdin.flush()

def read_response(proc, expect_id=None, timeout=TIMEOUT):
    start = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("No response from service (stdout closed)")
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if expect_id is None or msg.get("id") == expect_id:
            return msg
        if time.time() - start > timeout:
            raise TimeoutError("Timeout waiting for response")

def wait_for_status(proc, key, value=None, timeout=TIMEOUT):
    start = time.time()
    while True:
        line = proc.stderr.readline()
        if not line:
            raise RuntimeError("No status from service (stderr closed)")
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if key in msg and (value is None or msg[key] == value):
            return msg
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for status {key}")

def test_service():
    print(f"Launching: {EXE_PATH}")
    proc = subprocess.Popen(
        [str(EXE_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    try:
        # Wait for ready status
        status = wait_for_status(proc, "status", "ready", timeout=30)
        print("[OK] Service ready:", status)
        models_dir = status.get("models_dir")
        assert models_dir, "models_dir missing in status"
        print(f"[INFO] Models directory: {models_dir}")

        # List available models
        send_cmd(proc, {"cmd": "list_available_models", "id": "list1"})
        resp = read_response(proc, expect_id="list1")
        assert "models" in resp, "No models in list_available_models response"
        print(f"[OK] list_available_models: {len(resp['models'])} models")

        # For each test pair: download, translate, delete
        for src, tgt, test_text in TEST_PAIRS:
            lang_pair = f"{src}-{tgt}"
            print(f"\n[TEST] {lang_pair}: '{test_text}'")
            # Download model
            send_cmd(proc, {"cmd": "download_model", "id": f"dl_{lang_pair}", "lang_pair": lang_pair})
            # Wait for download progress and completion
            while True:
                status_msg = wait_for_status(proc, "download_progress", timeout=TIMEOUT)
                progress = status_msg["download_progress"]
                print(f"  [progress] {progress}")
                if progress.get("status") == "complete":
                    break
                if progress.get("status") == "error":
                    raise RuntimeError(f"Download error: {progress.get('error')}")
            # Confirm download result
            resp = read_response(proc, expect_id=f"dl_{lang_pair}")
            assert resp.get("success"), f"Download failed: {resp}"
            print("  [OK] Model downloaded and loaded")

            # Translate
            send_cmd(proc, {"cmd": "translate", "id": f"tr_{lang_pair}", "source": src, "target": tgt, "text": test_text})
            resp = read_response(proc, expect_id=f"tr_{lang_pair}")
            assert "translated" in resp, f"No translation in response: {resp}"
            print(f"  [OK] Translation: {resp['translated']}")

            # Delete model
            send_cmd(proc, {"cmd": "delete_model", "id": f"del_{lang_pair}", "lang_pair": lang_pair})
            resp = read_response(proc, expect_id=f"del_{lang_pair}")
            assert resp.get("success"), f"Delete failed: {resp}"
            print("  [OK] Model deleted")

        # Health check
        send_cmd(proc, {"cmd": "health", "id": "health1"})
        resp = read_response(proc, expect_id="health1")
        assert resp.get("status") == "ok", f"Health check failed: {resp}"
        print("[OK] Health check passed")

        # Shutdown
        send_cmd(proc, {"cmd": "shutdown", "id": "shutdown1"})
        print("[OK] Sent shutdown command")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

if __name__ == "__main__":
    test_service()
