# EAHTM 与补实验说明

本目录是论文主模型 **EAHTM（层次主题模型）** 与 **审稿补实验脚本** 的说明入口；**可执行代码与数据**在子目录 **`EAHTM/`**（即 `EAHTM/EAHTM/`）内。克隆 [Additional-experiments](https://github.com/liulejun511/Additional-experiments) 后，请进入 **`EAHTM/EAHTM`** 再运行下方命令。

---

## 1. 目录结构（你需要关心的）

| 路径 | 含义 |
|------|------|
| **`EAHTM/EAHTM/`** | 代码根目录：`run_HTM.py`、`configs/`、`models/`、`runners/`、`utils/` |
| **`EAHTM/EAHTM/data/<数据集名>/`** | 预处理数据：`train_bow.npz`、`test_bow.npz`、`vocab.txt`、`train_texts.txt`（部分见下）、`word_embeddings.npz` 等 |
| **`EAHTM/EAHTM/experiments/`** | 补实验：NMF 基线、Sinkhorn 扫描、collapse 诊断、CTM 数据准备、fastText 词向量构建、`run_suite` 统一入口 |
| **`EAHTM/EAHTM/output/`** | 训练产物（默认不提交 Git；本地生成） |

仓库根下另有 **`TopMost-main/`**（仅库代码，用于 ProdLDA / SawETM 等基线）、**`external_models/contextualized-topic-models/`**（**CTM / CombinedTM** 官方实现，与 [ACL 2021 短文](https://aclanthology.org/2021.acl-short.96/) 对应；说明见 **`external_models/README.md`**）。

---

## 2. 环境与依赖

建议使用虚拟环境，在 **`EAHTM/EAHTM`** 下安装：

```text
torch（与 CUDA 匹配）
pyyaml
gensim
scipy
scikit-learn
tqdm
```

版本可按你复现实验时的环境锁定；内层 **`EAHTM/EAHTM/requirements.txt`** 面向 EAHTM 训练与评估；若从仓库根一键安装（含 CTM 等），见根目录 **`requirements.txt`**。

---

## 3. 数据集与 Git 策略

| 数据集 | 仓库内 | 说明 |
|--------|--------|------|
| **20NG / NYT / ACL** | 主要文件在库内 | ACL **`train_texts.txt`** 体积常超 GitHub 单文件 100MB，**不入库**；本地放入 `data/ACL/train_texts.txt` 后再跑 C_V 等 |
| **NeurIPS** | bow、词向量、测试文本等在库内；**`train_texts.txt`** 常 >100MB | 不入库，本地放入 `data/NeurIPS/train_texts.txt` |
| **IMDB** | **不入库** | 需要时在本地创建 `data/IMDB/` 并放入与 20NG 同结构的文件 |

NeurIPS 文件命名与布局见 **`EAHTM/EAHTM/data/NeurIPS/README.txt`**。

---

## 4. 训练 EAHTM（主实验）

在 **`EAHTM/EAHTM`** 下：

```bash
# 基本训练（数据集名：20NG, NYT, ACL, NeurIPS, IMDB（仅本地有数据时））
python run_HTM.py -d 20NG --data_dir ./data -k 10-50-200 --test_index 1

# 可选：覆盖 Sinkhorn 温度、替换词向量、记录耗时与显存、记录 OT 熵/稀疏度
python run_HTM.py -d 20NG --data_dir ./data --sinkhorn_alpha 20 --word_embeddings_npz ./data/20NG/word_embeddings.fasttext.npz --log_training_stats --log_ot_stats
```

产出前缀：`output/<数据集>/HTM_K<层主题数>_<test_index>th`，含 `_T15` 主题词、`_params.npz`、`_embeddings.npz`、（若开启）`_training_stats.json`。

---

## 5. 离线评估（C_V、TD、TU、层次指标、下游）

仍在 **`EAHTM/EAHTM`** 下，`--path` 为上述前缀（**不要**带 `_T15` 后缀）：

```bash
python utils/eva/hierarchical_topic_quality.py --path output/20NG/HTM_K10-50-200_1th --dataset 20NG --data_dir ./data --num_top_words 15 --read_labels True
```

- 打印中 **C_V** 为主题一致性；**TD / TU** 为多样性相关；**CLNPMI、PC_TD** 等为层次父子关系指标。  
- **无 `phi_list` 的扁平基线**（如 NMF 输出）：自动跳过层次指标；必要时加 **`--skip_hierarchy`**。

---

## 6. 补实验脚本（与论文表格对应）

在 **`EAHTM/EAHTM`** 下优先使用统一入口：

```bash
python -m experiments.run_suite --help
```

常用子命令示例：

| 目的 | 示例 |
|------|------|
| **sklearn NMF 基线**（TF-IDF + NMF，非神经） | `python -m experiments.run_suite nmf --dataset 20NG` |
| **collapse 数值诊断** | `python -m experiments.run_suite collapse --path output/20NG/HTM_K10-50-200_1th --dataset 20NG` |
| **Sinkhorn α 扫描** | `python -m experiments.run_suite sinkhorn --dataset 20NG --alphas 5 10 20 40 --log_training_stats` |
| **层次树 JSON（定性）** | `python -m experiments.run_suite qualitative --path output/20NG/HTM_K10-50-200_1th` |
| **CTM 侧数据准备** | `python -m experiments.run_suite ctm_prep --dataset 20NG` |
| **fastText 词向量矩阵** | `python -m experiments.run_suite fasttext_prep --dataset 20NG --vectors <本机绝对路径> --binary`（`.vec` 则去掉 `--binary`） |
| **TopMost 效率对齐** | `python -m experiments.run_suite topmost --topmost_root <TopMost 克隆目录>` |

### 6.1 七项是否「全跑通」的含义

- **前六项**只依赖：数据集在 `data/<数据集>/`、（`collapse` / `qualitative`）已有一次 **`run_HTM.py`** 写出的 **`output/.../HTM_..._<n>th`** 前缀、`topmost` 依赖本机 TopMost 路径。
- **第七项 `fasttext_prep`** 额外依赖：你已**从官方下载并解压**的 **`cc.en.300.bin` 或 `.vec`**，且命令里 **`--vectors` 必须是真实路径**（文档里的 `D:\path\to\...` 仅为占位，不存在时会 `FileNotFoundError`）。
- 因此：其余六项已能跑通时，**补齐本机 fastText 文件并正确传路径**，七个子命令的**代码入口**即可全部执行成功。若再用生成的 npz 做 **EAHTM 训练对比**，需注意 **向量维度与 `configs/HTM.yaml` 中 `model.embedding_dim` 一致**（`cc.en.300` 为 300 维，默认配置常为 200，需二选一调整）。

### 6.2 fastText 预训练向量从哪获取

- 官方说明与下载列表：[fastText — Word vectors for 157 languages (crawl)](https://fasttext.cc/docs/en/crawl-vectors.html)。英语常用 **`cc.en.300.bin.gz`** 或 **`cc.en.300.vec.gz`**，解压后得到 `.bin` 或 `.vec`。
- **推荐 `.bin`**：体积相对小，`gensim` 加载时加 **`--binary`**（`run_suite fasttext_prep` 同理）。
- 默认输出：`data/<数据集>/word_embeddings.fasttext.npz`。若需自定义输出路径，请直接调用 **`python -m experiments.build_fasttext_embeddings ... --output <路径>`**（`run_suite fasttext_prep` 当前不转发 `--output`）。

### 6.3 可选：快速冒烟（非正式实验）

内层 **`configs/HTM_smoke.yaml`** 为 **1 个 epoch**、较小 OT 迭代，用于本机或协作者验证「训练 → 写文件 → `sinkhorn_sweep` 子进程」是否通畅；正式论文数值请仍用 **`HTM.yaml`** 与 README §4 的完整训练命令。

```bash
python run_HTM.py -d 20NG --data_dir ./data -m HTM_smoke -k 2-4-8 --test_index 1 --log_training_stats
python -m experiments.sinkhorn_sweep --dataset 20NG --data_dir ./data -k 2-4-8 --test_index 2 --alphas 20 --model_config HTM_smoke --log_training_stats
```

**说明**：不需要再单独引入已删除的顶层 **`NMF/`**（HyHTM）目录；NMF 经典基线用 **`experiments/nmf_baseline.py`**（经 `run_suite nmf`）即可。TopMost 内 **`NMF_trainer`** 属于另一套训练管线，需按 TopMost 的数据目录使用。

---

## 7. CTM baseline（上下文向量 + 主题模型）

回应审稿人对 **BERT / contextual embedding** 类基线的质疑：使用 **CombinedTM** 等模型，与论文 **[Bianchi et al., ACL-IJCNLP 2021](https://aclanthology.org/2021.acl-short.96/)** 一致；代码来源 **[MilaNLProc/contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)**，已 vendored 至仓库根 **`external_models/contextualized-topic-models/`**（更新方式见 **`external_models/README.md`**）。

**最低实验建议**：在 **2–4 个数据集**（如 20NG、NYT、ACL、NeurIPS）上跑通 CombinedTM，报告与主文一致的 **C_V、TD、TU**（本仓库评估脚本中 **TU** 对应 `hierarchical_topic_quality.py` 打印的 `TU`；CTM 导出单层 top words 后可用 **`--skip_hierarchy`** 只算 C_V/TD/TU）。

**推荐流程**：

1. **安装 CTM 包**（在仓库根或 venv 内）  
   `pip install -e external_models/contextualized-topic-models`  
   并安装 **`sentence-transformers`**、**`torch`**（版本按官方 README 与 CUDA 自选）。

2. **准备 BoW + 原文**（与 EAHTM `data/<数据集>` 对齐）  
   在 **`EAHTM/EAHTM`** 下：  
   `python -m experiments.ctm_baseline -d 20NG --data_dir ./data`  
   会在 `output/<数据集>/CTM_prep_*` 写出 BoW 的 csr 分量与 `train_texts_copy.txt` 等，供 CombinedTM 官方 notebook / 脚本读取。

3. **训练与导出主题词**  
   按上游仓库 README 使用 `CombinedTM`；将得到的 top words 存成与 EAHTM 相同行首格式（如 `L-0_K-0 w1 w2 ...`），再用 **`utils/eva/hierarchical_topic_quality.py`** 计算 C_V、TD、TU（必要时 **`--skip_hierarchy`**）。

---

## 8. 与仓库根目录的关系

- 本文件路径：`Additional-experiments/EAHTM/README.md`  
- **实际运行目录**：`Additional-experiments/EAHTM/EAHTM/`  

若在 IDE 中从外层 `EAHTM` 打开，请再进入一层 **`EAHTM`** 执行 `python run_HTM.py ...`，或将工作目录设为内层 `EAHTM`。

---

## 9. 推送到 GitHub

在仓库根 **`Additional-experiments/`**（与 `.git` 同级）执行：

```bash
git add EAHTM/README.md EAHTM/EAHTM/...
git commit -m "docs: EAHTM README and comments"
git push origin main
```

具体路径以你本地改动为准。
