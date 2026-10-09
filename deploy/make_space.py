#!/usr/bin/env python3
"""
deploy/make_space.py
====================
Builds deploy/space/: a clean, public-safe copy of the nursery for hosting where anyone can raise
their own baby. Your saves, training data, story-reading checkpoint and analysis stay private.

Two kinds of build:
  static (default, free): the brain runs inside each visitor's browser (web/baby.onnx, made by
                          deploy/export_web.py). Works on a free Hugging Face static Space or GitHub
                          Pages, for any number of visitors, and chats never leave their device.
  --docker              : the brain runs on a server (app.py --public). Needs a paid Space or
                          another Docker host.

  python deploy/make_space.py --push YOUR_NAME/baby-llm   # free static Space (log in once: hf auth login)
  python deploy/make_space.py --docker                     # build the server version
  docker build -t baby-llm deploy/space && docker run -p 7860:7860 baby-llm
"""

import os
import sys
import shutil
import argparse

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deploy", "space")

STATIC_FILES = ["index.html", "tokenizer/tokenizer.json", ".nojekyll"]
STATIC_DIRS = ["css", "js", "sounds", "web"]
DOCKER_FILES = ["app.py", "chat.py", "train.py", "sft_data.py", "index.html",
                "tokenizer/tokenizer.json", "tokenizer/vocab.json", "checkpoints/baby_chat.pt"]
DOCKER_DIRS = ["css", "js", "sounds"]

DOCKERFILE = """\
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
# CPU-only PyTorch keeps the image small; the baby's brain is tiny
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \\
 && pip install tokenizers==0.23.2
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["python", "app.py", "--public"]
"""

SPACE_README = """\
---
title: Pip, a baby LLM to raise
emoji: 👶
colorFrom: pink
colorTo: yellow
sdk: {sdk}
{port_line}pinned: false
short_description: Raise a tiny language model that learns from you
---

# Pip: a baby LLM to raise

Name a baby, feed it, rock it to sleep, dress it up, and teach it about the world. It's a
17-million-parameter language model trained from scratch on children's stories. It doesn't
memorize: it was trained to reason over what *you* teach it, and it says when it doesn't know.

- **Your baby is yours.** It lives in your browser, and {privacy}
  Use 💾 to save a backup file and 📂 to load it on another device.
- **Be kind.** Babies don't learn rude words here.
- **Click once after the page loads** to let the baby make sounds.

Sounds: Pixabay contributors (see `sounds/CREDITS.md`). Stories: TinyStories (Eldan & Li, 2023).
"""


def build(docker: bool = False):
    if not docker and not os.path.exists(os.path.join(ROOT, "web", "baby.onnx")):
        sys.exit("Missing web/baby.onnx. Build it first: python deploy/export_web.py (in the onnx environment)")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    files, dirs = (DOCKER_FILES, DOCKER_DIRS) if docker else (STATIC_FILES, STATIC_DIRS)
    for f in files:
        src = os.path.join(ROOT, f)
        if not os.path.exists(src):
            sys.exit(f"Missing {f}. Train the baby first (python sft.py).")
        dst = os.path.join(OUT, f)
        os.makedirs(os.path.dirname(dst) or OUT, exist_ok=True)
        shutil.copy2(src, dst)
    for d in dirs:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(OUT, d))
    if docker:
        with open(os.path.join(OUT, "Dockerfile"), "w", encoding="utf-8") as fh:
            fh.write(DOCKERFILE)
        with open(os.path.join(OUT, ".dockerignore"), "w", encoding="utf-8") as fh:
            fh.write("__pycache__\n*.pyc\n")
    readme = SPACE_README.format(
        sdk="docker" if docker else "static",
        port_line="app_port: 7860\n" if docker else "",
        privacy=("what you say is only used to make the baby's reply and isn't stored on the server."
                 if docker else "its brain runs right on your device: nothing you say is sent anywhere."),
    )
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"[✓] Built {'server (Docker)' if docker else 'static'} version in {OUT} ({size / 2**20:.0f} MB)")


def push(repo_id: str, docker: bool = False):
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()  # fails early with a clear message if not logged in
    api.create_repo(repo_id, repo_type="space", space_sdk="docker" if docker else "static", exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=repo_id, repo_type="space", commit_message="Deploy the baby nursery")
    print(f"[✓] Uploaded. Your nursery will be live shortly at https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build (and optionally publish) the public nursery")
    ap.add_argument("--push", metavar="USER/SPACE", help="Upload to this Hugging Face Space after building")
    ap.add_argument("--docker", action="store_true", help="Server version (paid Space or other host) instead of the free static one")
    args = ap.parse_args()
    build(args.docker)
    if args.push:
        push(args.push, args.docker)
