NeurIPS 论文语料（EAHTM 用）
============================

本目录需与 `../20NG/` 相同的一批文件（名称一致即可被 `run_HTM.py -d NeurIPS` 读取）：

  train_bow.npz      test_bow.npz
  train_texts.txt    test_texts.txt
  vocab.txt          word_embeddings.npz

可选（仅当要跑 hierarchical_topic_quality.py 的聚类/分类且脚本要求标签时）：

  train_labels.txt   test_labels.txt

无标签时：训练 EAHTM 可不提供 labels；评估阶段对 NeurIPS 请使用 `--read_labels False`，或自行生成占位标签文件。

数据获取参考
------------
- TopMost 仓库提供同名语料压缩包与下载逻辑，见：
  https://github.com/bobxwu/topmost
  解压或运行其 `download_dataset('NeurIPS', ...)` 后，需将格式转换为上述 scipy.sparse 的 bow npz + 每行一篇的 txt（与当前 20NG 预处理一致）。

规模参考（与 TopMost 测试一致）：约 7237 篇、词表约 10000。
