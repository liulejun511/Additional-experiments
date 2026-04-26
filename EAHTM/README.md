## 本仓库（GitHub）范围说明

纳入版本控制的主要包括：**EAHTM 主实验代码与数据集 20NG / NYT / ACL** 的预处理文件，以及 **NeurIPS** 目录下的 `README.txt`（说明需自备的 bow/文本/词向量文件名与获取方式）；NeurIPS 实际数据请按该说明放入 `EAHTM/EAHTM/data/NeurIPS/`。**IMDB** 暂不纳入 Git，需要时在本地创建 `data/IMDB/`。**补实验脚本**见 `EAHTM/EAHTM/experiments/`。**TopMost** 仅保留 **Python 库与 `run.py` 等**，`data/` 与 `output/` 不入库。ACL、NeurIPS 的 `train_texts.txt` 因超过 GitHub 单文件上限，不入库；需自行放入 `EAHTM/EAHTM/data/ACL/`、`data/NeurIPS/`。

## Usage

### 1. Environment Setup
Make sure to install the following packages (recommended using a virtual environment):

    python==3.8.0
    pytorch==1.7.1
    gensim==4.3.0
    scipy==1.5.2
    scikit-learn==0.24.2
    tqdm
    pyyaml


### 2. Train and evaluate the model
在 `EAHTM/EAHTM` 目录下执行，例如：

    python run_HTM.py -d 20NG --data_dir ./data
    python utils/eva/hierarchical_topic_quality.py --path output/20NG/HTM_K10-50-200_1th --dataset 20NG --data_dir ./data


