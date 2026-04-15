"""Lightweight demo server for the Laplace Transformer.

Runs on stdlib only (no Flask / FastAPI needed). Binds to 0.0.0.0 so your
phone on the same Wi-Fi network can load the page. On startup we train a
tiny copy-task model so the /api/copy endpoint can show off in-context
pattern matching.

    python web/server.py --port 8000

Then point your phone browser at http://<your-laptop-lan-ip>:<port>/.
The server prints candidate URLs on startup.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

# Let `python web/server.py` work from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace_transformer import CharTokenizer, LaplaceConfig, LaplaceTransformer


WEB_DIR = Path(__file__).resolve().parent
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-"
SEP = "|"


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------
class DemoState:
    """Holds the two models we serve: a trained copy-model and a fresh
    matcher. Protected by a lock because the HTTP server is threaded."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

        # ---- Copy-task model: trained to echo a pattern after `|` ----
        self.copy_tok = CharTokenizer.from_text(ALPHABET + SEP)
        self.copy_pat_len = 8
        self.copy_cfg = LaplaceConfig(
            vocab_size=self.copy_tok.vocab_size,
            max_len=2 * self.copy_pat_len + 4,
            d_model=96,
            num_heads=4,
            num_layers=2,
            ffn_mult=4,
            dropout=0.0,
            causal=True,
            use_feature_map=False,
            pos_num_modes=min(16, 2 * self.copy_pat_len + 4),
        )
        self.copy_model = LaplaceTransformer(self.copy_cfg).eval()

        # ---- Matcher model: untrained; Laplace attention is already sharp ----
        self.match_tok = CharTokenizer.from_text(ALPHABET)
        self.match_cfg = LaplaceConfig(
            vocab_size=self.match_tok.vocab_size,
            max_len=512,
            d_model=64,
            num_heads=4,
            num_layers=2,
            pos_num_modes=32,
            use_feature_map=False,
        )
        self.match_model = LaplaceTransformer(self.match_cfg).eval()

    def train_copy_model(self, steps: int = 400, batch_size: int = 64) -> None:
        """Train the copy model on random `pat|pat` sequences."""
        model = self.copy_model.train()
        tok = self.copy_tok
        pat_len = self.copy_pat_len
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        alpha = "".join(c for c in ALPHABET if c not in " .,!?-")  # letters+digits only
        for step in range(1, steps + 1):
            xs, ys = [], []
            for _ in range(batch_size):
                p = "".join(random.choice(alpha) for _ in range(pat_len))
                ids = tok.encode(p + SEP + p)
                xs.append(ids[:-1])
                y = list(ids[1:])
                y[:pat_len] = [-100] * pat_len
                ys.append(y)
            x = torch.tensor(xs)
            y = torch.tensor(ys)
            out = model(x, targets=y)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0 or step == 1:
                print(f"[copy-model] step {step:4d}  loss {out['loss'].item():.4f}")
        model.eval()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    state: DemoState = None  # set by main()

    # ---------- helpers ----------
    def _send_json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):  # quieter default logs
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    # ---------- routing ----------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/style.css":
            self._send_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        elif self.path == "/api/health":
            self._send_json({"ok": True, "device": "cpu"})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        try:
            if self.path == "/api/match":
                self._handle_match()
            elif self.path == "/api/copy":
                self._handle_copy()
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # keep the server alive on bad input
            self._send_json({"error": str(exc)}, status=400)

    # ---------- API handlers ----------
    def _handle_match(self) -> None:
        body = self._read_json()
        haystack = (body.get("haystack") or "").lower()
        needle = (body.get("needle") or "").lower()
        layer = int(body.get("layer", -1))
        if not haystack or not needle:
            raise ValueError("Both 'haystack' and 'needle' are required")
        state = self.state
        tok = state.match_tok
        # Replace any unknown characters with space so the demo never crashes.
        haystack = "".join(c if c in tok.stoi else " " for c in haystack)
        needle = "".join(c if c in tok.stoi else " " for c in needle)

        if len(haystack) + len(needle) > state.match_cfg.max_len:
            raise ValueError("Input too long for demo model (max 512 chars total)")

        with state.lock:
            scores = state.match_model.match_pattern(
                tok.encode_tensor(haystack),
                tok.encode_tensor(needle),
                layer=layer,
            )
        self._send_json(
            {
                "haystack": haystack,
                "needle": needle,
                "scores": [float(s) for s in scores.tolist()],
            }
        )

    def _handle_copy(self) -> None:
        body = self._read_json()
        pattern = (body.get("pattern") or "").lower()
        state = self.state
        tok = state.copy_tok
        pat_len = state.copy_pat_len
        allowed = "".join(c for c in ALPHABET if c not in " .,!?-")
        pattern = "".join(c for c in pattern if c in allowed)
        if not pattern:
            raise ValueError("Pattern must contain letters or digits")
        if len(pattern) > pat_len:
            pattern = pattern[:pat_len]
        prompt = pattern + SEP

        with state.lock:
            ids = tok.encode_tensor(prompt).unsqueeze(0)
            out = state.copy_model.generate(
                ids, max_new_tokens=len(pattern), temperature=1e-6
            )
        full = tok.decode(out[0].tolist())
        completion = full[len(prompt):] if full.startswith(prompt) else full[-len(pattern):]
        self._send_json(
            {
                "prompt": prompt,
                "completion": completion,
                "correct": completion == pattern,
            }
        )


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def guess_lan_ips() -> list[str]:
    """Best-effort list of LAN IPs to advertise to the user."""
    ips = set()
    try:
        # Trick: open a UDP socket to an external address — no traffic sent,
        # but the kernel picks the default outbound IP.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ":" in ip:
                continue  # skip IPv6 for simplicity
            if ip.startswith("127."):
                continue
            ips.add(ip)
    except socket.gaierror:
        pass
    return sorted(ips)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--train-steps", type=int, default=400,
                        help="Copy-task training steps on startup (set 0 to skip).")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    print("Loading models...")
    state = DemoState()
    if args.train_steps > 0:
        print(f"Training copy model for {args.train_steps} steps (CPU)...")
        state.train_copy_model(steps=args.train_steps)
    Handler.state = state

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("\n--- Laplace Transformer demo ---")
    print(f"Listening on {args.host}:{args.port}")
    print("Open on this device:  http://localhost:{}/".format(args.port))
    for ip in guess_lan_ips():
        print(f"Open on your phone:   http://{ip}:{args.port}/")
    print("(Phone must be on the same Wi-Fi network.)")
    print("Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
