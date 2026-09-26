"""CLI: ``python -m sglang_omni_backend --help``."""

from __future__ import annotations

import argparse
import logging
import os

from aiohttp import web

from .server import create_app
from .session import BackendConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o demo backend served by a sglang-omni /v1/realtime server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22500)
    parser.add_argument(
        "--upstream-url",
        default=os.environ.get("UPSTREAM_URL", "ws://127.0.0.1:18260/v1/realtime"),
        help="sglang-omni /v1/realtime URL (env UPSTREAM_URL)",
    )
    parser.add_argument("--max-sessions", type=int, default=int(os.environ.get("MAX_SESSIONS", "1")),
                        help="concurrent sessions in this process (the reference backend allows 1); each needs one worker")
    parser.add_argument("--forward-images", action=argparse.BooleanOptionalAction, default=True,
                        help="forward per-unit video frames as sglang.input_image.append when the server grants image input")
    parser.add_argument("--silence-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="advance the model with silent units when the worker stops sending (e.g. page paused)")
    parser.add_argument("--silence-grace-ms", type=float, default=1000.0,
                        help="how late the next chunk may be before silence is filled in")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = BackendConfig(
        upstream_url=args.upstream_url,
        forward_images=args.forward_images,
        silence_fill=args.silence_fill,
        silence_grace_ms=args.silence_grace_ms,
    )
    web.run_app(create_app(config, max_sessions=args.max_sessions), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
