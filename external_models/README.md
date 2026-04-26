# external_models

## `contextualized-topic-models`（CTM / CombinedTM 官方实现）

对应论文：**[Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence](https://aclanthology.org/2021.acl-short.96/)**（ACL-IJCNLP 2021 Short）。

- **上游仓库**：[MilaNLProc/contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)

本仓库中的副本为 **vendored 源码树**（不含嵌套 `.git`），与主仓库一并版本管理，便于审稿复现与单一 `git push`。

### 在本机更新到最新上游

网络正常时在仓库根目录执行（PowerShell 可自行改成 `Remove-Item`）：

```bash
rm -rf external_models/contextualized-topic-models
git clone https://github.com/MilaNLProc/contextualized-topic-models.git external_models/contextualized-topic-models
rm -rf external_models/contextualized-topic-models/.git
git add external_models/contextualized-topic-models
git commit -m "chore: refresh contextualized-topic-models from upstream"
```

### 安装到当前 Python 环境

```bash
pip install -e external_models/contextualized-topic-models
pip install sentence-transformers torch
```

训练 CombinedTM 前，EAHTM 侧可用 `EAHTM/EAHTM/experiments/ctm_baseline.py` 生成与 `data/<数据集>` 对齐的 BoW 与文本副本（见 `EAHTM/README.md` 补实验一节）。
