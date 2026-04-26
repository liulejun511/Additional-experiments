#!/usr/bin/env python3
"""自检 requirements.txt 环境：导入关键包并做一次极小计算。"""
from __future__ import annotations

import importlib
import sys
from typing import List, Tuple


def _try(mod: str, attr: str | None = "__version__") -> Tuple[str, str]:
    m = importlib.import_module(mod)
    ver = getattr(m, attr, "") if attr else ""
    if callable(ver):
        ver = str(ver())
    elif ver:
        ver = str(ver)
    else:
        ver = "(no __version__)"
    return mod, ver


def main() -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except Exception:
                pass

    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print()

    modules: List[str] = [
        "numpy",
        "scipy",
        "yaml",
        "gensim",
        "sklearn",
        "tqdm",
        "pandas",
        "torch",
        "torchvision",
        "sentence_transformers",
        "transformers",
        "accelerate",
        "matplotlib",
        "nltk",
        "wordcloud",
        "contextualized_topic_models",
    ]

    failed = False
    for name in modules:
        try:
            mod, ver = _try(name)
            print(f"  OK  {mod:32s} {ver}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name:32s} {e!r}")
            failed = True

    if failed:
        print("\n有模块导入失败，请检查 venv 与 pip install -r requirements.txt。")
        return 1

    import torch

    x = torch.randn(4, 4, device="cpu")
    y = (x @ x.T).mean().item()
    print()
    print(f"Torch CPU 试算 mean(x @ x.T) = {y:.6f} （应为一有限浮点数）")

    import gensim

    _ = gensim.models.Word2Vec([["hello", "world"]], vector_size=8, min_count=1, epochs=1)
    print("Gensim Word2Vec 极小语料 1 epoch 完成。")

    print("\n环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
