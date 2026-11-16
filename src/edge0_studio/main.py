"""CLI entry point."""

from __future__ import annotations

import argparse
import logging


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="edge0-studio",
        description="Web UI and MCP bridge for Edge0 models served by edge0-infer",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    uvicorn.run("edge0_studio.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
