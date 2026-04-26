import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ======= 修改这里的路径 =======
npz_path = '//EAHTM/output/NeurIPS/HTM_embeddings.npz'

# ======= 加载嵌入向量 =======
data = np.load(npz_path, allow_pickle=True)
topic_embeddings_list = data['topic_embeddings_list']
word_embeddings = data['word_embeddings']


# 假设 word_embeddings 是 shape=(N, D)，N 是总词数，D 是维度
num_samples = 2000
total_words = word_embeddings.shape[0]

# 设置随机种子以保证可复现（可选）
np.random.seed(42)

# 随机选择不重复的 2000 个索引
sample_indices = np.random.choice(total_words, size=num_samples, replace=False)

# 获取对应的词嵌入
sampled_word_embeddings = word_embeddings[sample_indices]


# ======= 构造绘图数据 =======
all_embeddings = []
labels = []
colors = []

# 词嵌入
all_embeddings.append(sampled_word_embeddings)
labels.extend(['word'] * len(sampled_word_embeddings))
colors.extend(['black'] * len(sampled_word_embeddings))

# 每一层的主题嵌入
cmap = plt.get_cmap('tab10')
# for layer_id, topic_emb in enumerate(topic_embeddings_list):
#     all_embeddings.append(topic_emb)
#     labels.extend([f'topic_L{layer_id}'] * len(topic_emb))
#     colors.extend([cmap(layer_id)] * len(topic_emb))


all_embeddings.append(topic_embeddings_list[-1])
labels.extend([f'bottom_topic'] * len(topic_embeddings_list[-1]))
colors.extend([cmap(0)] * len(topic_embeddings_list[-1]))

all_embeddings = np.vstack(all_embeddings)

# ======= t-SNE 降维 =======
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

# # ======= 可视化 =======
# plt.figure(figsize=(10, 8))
# unique_labels = list(set(labels))
# for label in unique_labels:
#     indices = [i for i, l in enumerate(labels) if l == label]
#     plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
#                 label=label, alpha=0.6, s=20)
#
# plt.legend()
# plt.title("t-SNE of Topic and Word Embeddings")
# plt.show()


# ======= 可视化 =======
plt.figure(figsize=(6, 6))

# 词嵌入：蓝色圆点
word_indices = [i for i, label in enumerate(labels) if label == 'word']
plt.scatter(embeddings_2d[word_indices, 0], embeddings_2d[word_indices, 1],
            c='blue', marker='o', label='Word Embedding', alpha=0.6, s=5)

# 最后一层主题嵌入：黑色三角形
topic_indices = [i for i, label in enumerate(labels) if label == 'bottom_topic']
plt.scatter(embeddings_2d[topic_indices, 0], embeddings_2d[topic_indices, 1],
            c='red', marker='^', label='Bottom-level Topic Embedding', alpha=0.8, s=10)

# 去除刻度线和刻度数字，但保留坐标轴
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.savefig("AHTM5_embedding.pdf",dpi=300,bbox_inches='tight')
plt.show()
