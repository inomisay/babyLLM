#!/usr/bin/env python3
"""
grow.py
=======
One command to help the baby grow, safely:

  1. Read more stories (stream_train.py): learns language from TinyStories (~450k stories by default), streamed from
     Hugging Face and never stored. Continues from the current brain; pause with Ctrl+C and run
     grow.py again to resume.
  2. Practice conversations (sft.py): reasoning over memory, staying consistent, following up,
     generated fresh so nothing is memorized.
  3. Health check: the new brain is tested against the current one on the same held-out exam
     (words it never trained on) and only replaces it if it does better.

Usage:
  python grow.py                    # ~13 h of reading + ~1.5 h of conversation practice
  python grow.py --tokens 300M      # read longer
  python grow.py --skip-reading     # only redo the conversation practice (~1 h)
"""

import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Help the baby grow (resumable, never makes it worse)")
    ap.add_argument("--tokens", default="100M", help="How much story text to read, e.g. 100M (~13 h on CPU), 300M")
    ap.add_argument("--sft-steps", type=int, default=4000, help="Conversation practice steps (~1.3 s each on CPU)")
    ap.add_argument("--skip-reading", action="store_true", help="Skip step 1")
    args = ap.parse_args()
    os.chdir(HERE)

    if not args.skip_reading:
        print("=" * 72 + "\n  STEP 1/2: reading stories (streamed, nothing stored)\n" + "=" * 72)
        from stream_train import train_stream
        if train_stream(tokens=args.tokens) is False:
            print("\nPaused. Run `python grow.py` again to continue reading where it stopped.")
            return

    print("\n" + "=" * 72 + "\n  STEP 2/2: conversation practice + health check\n" + "=" * 72)
    cmd = [sys.executable, "sft.py", "--init", "checkpoints/baby_model_best.pt", "--out", "checkpoints/baby_chat.pt",
           "--keep-better", "--steps", str(args.sft_steps)]
    code = subprocess.call(cmd)
    if code == 0:
        print("\nAll done. Restart `python app.py` to talk with the grown-up baby.")
    else:
        print("\nConversation practice stopped early; the current brain was not changed.")


if __name__ == "__main__":
    main()
