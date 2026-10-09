#!/usr/bin/env python3
"""
app.py
======
Serves the Baby LLM nursery (index.html, js/, css/) and the /api/chat endpoint.
Uses checkpoints/baby_chat.pt (from sft.py) when present, else the pretrained story model.

Usage:
  python app.py
  python app.py --port 8765
"""

from chat import run_http_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baby LLM Web App Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (default: auto)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Tokenizer file")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the nursery in a browser")
    parser.add_argument("--public", action="store_true",
                        help="Host for everyone: each visitor's baby lives in their own browser, nothing is stored "
                             "on the server, replies are rate-limited. Listens on 0.0.0.0 and $PORT (default 7860)")
    args = parser.parse_args()
    if args.public:
        import os
        args.host = "0.0.0.0" if args.host == "127.0.0.1" else args.host
        args.port = int(os.environ.get("PORT", 7860)) if args.port == 8765 else args.port
        args.no_browser = True

    run_http_server(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        public=args.public,
    )
