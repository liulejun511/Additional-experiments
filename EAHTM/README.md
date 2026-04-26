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

仓库根下另有 **`TopMost-main/`**（仅库代码，用于 ProdLDA / SawETM 等基线）、**`external_models/`**（合并远程时带入的 CTM 相关代码，按需使用）。

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

版本可按你复现实验时的环境锁定；`requirements.txt` 在 `EAHTM/EAHTM/requirements.txt`。

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
| **fastText 词向量矩阵** | `python -m experiments.run_suite fasttext_prep --dataset 20NG --vectors <路径> --binary` |

**说明**：不需要再单独引入已删除的顶层 **`NMF/`**（HyHTM）目录；NMF 经典基线用 **`experiments/nmf_baseline.py`** 即可。TopMost 内 **`NMF_trainer`** 属于另一套训练管线，需按 TopMost 的数据目录使用。

---

## 7. 与仓库根目录的关系

- 本文件路径：`Additional-experiments/EAHTM/README.md`  
- **实际运行目录**：`Additional-experiments/EAHTM/EAHTM/`  

若在 IDE 中从外层 `EAHTM` 打开，请再进入一层 **`EAHTM`** 执行 `python run_HTM.py ...`，或将工作目录设为内层 `EAHTM`。

---

## 8. 推送到 GitHub

在仓库根 **`Additional-experiments/`**（与 `.git` 同级）执行：

```bash
git add EAHTM/README.md EAHTM/EAHTM/...
git commit -m "docs: EAHTM README and comments"
git push origin main
```

具体路径以你本地改动为准。
