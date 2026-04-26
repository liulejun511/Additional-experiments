from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import argparse
import numpy as np
import scipy.sparse
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re
import sys

sys.path.append('./')
from utils.data import file_utils
from utils.model import model_utils
from utils.eva import show_topic_hierarchy
from utils.eva  import clustering
from utils.eva  import classification



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--dataset')
    parser.add_argument('--num_top_words', default=15)
    parser.add_argument('--cv_type', default='c_v')
    parser.add_argument('--data_dir')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--read_labels', default=False)
    parser.add_argument(
        '--skip_hierarchy',
        action='store_true',
        help='Skip PCC/PCD/sibling metrics (e.g. NMF/CTM flat baselines without phi_list).',
    )
    args = parser.parse_args()
    return args


def read_all_bow(dataset_path):
    train_bow = scipy.sparse.load_npz(f'{dataset_path}/train_bow.npz').toarray().astype('float32')
    test_bow = scipy.sparse.load_npz(f'{dataset_path}/test_bow.npz').toarray().astype('float32')
    all_bow = np.concatenate((train_bow, test_bow), axis=0)
    return all_bow



def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = file_utils.split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = file_utils.split_text_word(reference_corpus)
    dictionary = Dictionary(file_utils.split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary, topics=split_top_words, topn=num_top_words, coherence=cv_type,processes=1)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score

def compute_TD(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD
def compute_topic_diversity(top_words, _type="TD"):
    TD = compute_TD(top_words)
    return TD


def multiaspect_topic_diversity(top_words, _type="TD"):
    TD_list = list()
    for level_top_words in top_words:
        TD = compute_topic_diversity(level_top_words, _type)
        TD_list.append(TD)

    return np.mean(TD_list)


def multiaspect_topic_uniqueness(top_words):
    TU_list = list()
    for level_top_words in top_words:
        TU_list.append(compute_TU(level_top_words))
    return np.mean(TU_list)

def compute_TU(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TU = (TF != 0).sum() / (K * T)

    return TU

def compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab):
    npmi_list = list()

    for p_w in parent_diff_words:
        flag_n = all_bow[:, vocab.index(p_w)] > 0
        p_n = np.sum(flag_n) / len(all_bow)

        for c_w in child_diff_words:
            flag_l = all_bow[:, vocab.index(c_w)] > 0
            p_l = np.sum(flag_l)
            p_nl = np.sum(flag_n * flag_l)

            if p_nl == len(all_bow):
                npmi_score = 1
            else:
                p_l = p_l / len(all_bow)
                p_nl = p_nl / len(all_bow)
                p_nl += 1e-10
                npmi_score = np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)

            npmi_list.append(npmi_score)

    return npmi_list


def get_CLNPMI(PC_pair_groups, all_bow, vocab):
    CNPMI_list = list()
    for group in tqdm(PC_pair_groups):
        layer_CNPMI = list()
        for parent_topic, child_topic in group:
            parent_words = set(parent_topic.split())
            child_words = set(child_topic.split())

            inter = parent_words.intersection(child_words)
            parent_diff_words = list(parent_words.difference(inter))
            child_diff_words = list(child_words.difference(inter))

            npmi_list = compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab)

            # NOTE: assign -1 to the NPMI of repetitive word pairs
            num_repetition = (len(parent_words) - len(parent_diff_words)) * (len(child_words) - len(child_diff_words))
            npmi_list.extend([-1] * num_repetition)

            layer_CNPMI.extend(npmi_list)

        CNPMI_list.append(np.mean(layer_CNPMI))

    return CNPMI_list


def compute_diff_topic_pair(topic_str_a, topic_str_b):
    word_counter = Counter()
    topic_words_a = topic_str_a.split()
    topic_words_b = topic_str_b.split()
    word_counter.update(topic_words_a)
    word_counter.update(topic_words_b)
    diff = (np.asarray(list(word_counter.values())) == 1).sum() / (len(topic_words_a) + len(topic_words_b))
    return diff


def get_topics_difference(topic_pair_groups):
    diff_list = list()
    for groups in topic_pair_groups:
        layer_diff = list()
        for topic_a, topic_b in groups:
            diff = compute_diff_topic_pair(topic_a, topic_b)
            layer_diff.append(diff)
        diff_list.append(np.mean(layer_diff))

    return diff_list


# Given a list of child topics, find the nonchild topics based on their item info.
def extract_nonchild_topic_list(hierarchical_topic_dict, child_topic_list, num_topics_list):
    child_topic_idx_list = [model_utils.parse_item_info(child_topic)[1] for child_topic in child_topic_list]
    layer_id, _ = model_utils.parse_item_info(child_topic_list[0])
    # num_topic = beta_list[layer_id].shape[0]
    num_topic = num_topics_list[layer_id]
    nonchild_topic_idx_list = list(set(range(num_topic)) - set(child_topic_idx_list))
    nonchild_topic_list = np.asarray(hierarchical_topic_dict[layer_id])[nonchild_topic_idx_list].tolist()

    return nonchild_topic_list


def get_topic_pairs(topic_pairs, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type, layer_id=0):
    for parent_topic in topic_hierarchy.keys():
        if isinstance(topic_hierarchy[parent_topic], list):
            child_topic_list = topic_hierarchy[parent_topic]
        else:
            child_topic_list = list(topic_hierarchy[parent_topic].keys())

        if _type == 'PC':
            for child_topic in child_topic_list:
                topic_pairs[layer_id].append((parent_topic, child_topic))

        elif _type == 'PnonC':
            nonchild_topic_list = extract_nonchild_topic_list(hierarchical_topic_dict, child_topic_list, num_topics_list)
            for nonchild_topic in nonchild_topic_list:
                topic_pairs[layer_id].append((parent_topic, nonchild_topic))

        # Move to the next layer if more.
        if not isinstance(topic_hierarchy[parent_topic], list):
            get_topic_pairs(topic_pairs, topic_hierarchy[parent_topic], hierarchical_topic_dict, num_topics_list, _type, layer_id + 1)


# sibling_groups: length == num_layers
# each element in the list is a group of sibling topics at a layer.
def get_sibling_groups(topic_hierarchy, sibling_groups, layer_id=0):
    if isinstance(topic_hierarchy, list):
        sibling_groups[layer_id].append(topic_hierarchy)
    else:
        # sibling topics at this layer
        sibling_groups[layer_id].append(list(topic_hierarchy.keys()))
        # sibling topics at next layer
        for parent_topic in topic_hierarchy.keys():
            get_sibling_groups(topic_hierarchy[parent_topic], sibling_groups, layer_id + 1)


def get_Sibling_TD(sibling_groups):
    sibling_TD = list()
    for group in sibling_groups:
        layer_sibling_TD = list()
        for sibling_topics in group:
            TD = compute_TD(sibling_topics)
            layer_sibling_TD.append(TD)
        sibling_TD.append(np.mean(layer_sibling_TD))
    return sibling_TD


def get_Sibling_NPMI(sibling_groups, all_bow, vocab):
    sibling_NPMI = list()
    for group in sibling_groups:
        layer_pairs = list()
        for sibling_topics in group:
            sibling_num = len(sibling_topics)
            for i in range(sibling_num):
                for j in range(i + 1, sibling_num):
                    layer_pairs.append([sibling_topics[i], sibling_topics[j]])

        npmi = get_CLNPMI(layer_pairs, all_bow, vocab)
        sibling_NPMI.append(np.mean(npmi))
    return sibling_NPMI


def get_topic_groups(hierarchical_topic_dict, beta_list, phi_list):
    topic_hierarchy = show_topic_hierarchy.build_hierarchy(hierarchical_topic_dict, phi_list)

    num_layers = len(beta_list)
    num_topics_list = [item.shape[0] for item in beta_list]

    PC_pair_groups = [list() for _ in range(num_layers - 1)]
    PnonC_pair_groups = [list() for _ in range(num_layers - 1)]

    get_topic_pairs(PC_pair_groups, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type='PC')
    get_topic_pairs(PnonC_pair_groups, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type='PnonC')

    sibling_groups = [list() for _ in range(num_layers)]
    get_sibling_groups(topic_hierarchy, sibling_groups)

    # Because hierarhcial_topic_dict contains item info of each topic (Layer-0_K-20)
    # Remove these item info from topic strings
    PC_pair_groups = clean_group_info(PC_pair_groups)
    PnonC_pair_groups = clean_group_info(PnonC_pair_groups)
    sibling_groups = clean_group_info(sibling_groups)
    return PC_pair_groups, PnonC_pair_groups, sibling_groups


def clean_group_info(groups):
    clean_groups = list()
    for layer_group in groups:
        layer_clean_group = list()
        for topic_str_list in layer_group:
            layer_clean_group.append(clean_info(topic_str_list))
        clean_groups.append(layer_clean_group)
    return clean_groups


# remove item info of topic strings
# L-0_K-0 w1 w2 ===> w1 w2.
def clean_info(topic_str_list):
    clean_list = [" ".join(item.split()[1:]) for item in topic_str_list]
    return clean_list


def main():
    args = parse_args()
    T = args.num_top_words
    read_labels = args.read_labels


    data_mat = np.load(f'{args.path}_params.npz', allow_pickle=True)
    beta_list = data_mat['beta_list']
    train_theta = data_mat['train_theta_list']
    test_theta = data_mat['test_theta_list']
    has_phi = 'phi_list' in data_mat.files
    run_hierarchy = has_phi and not args.skip_hierarchy


    dataset_path = f'{args.data_dir}/{args.dataset}'
    all_bow = read_all_bow(dataset_path)
    vocab = file_utils.read_text(f'{dataset_path}/vocab.txt')
    corpus = file_utils.read_text(f'{dataset_path}/train_texts.txt')
    top_words = file_utils.read_text(f'{args.path}_T{T}')


    processed_top_words = [
        re.sub(r'^L-\d+_K-\d+\s', '', item) if re.match(r'^L-\d+_K-\d+\s', item) else item
        for item in top_words
    ]

    layers = defaultdict(list)
    # 遍历每个主题项并解析层次信息
    for item in top_words:
        match = re.match(r'^L-(\d+)_K-\d+\s', item)
        if match:
            layer = int(match.group(1))  # 获取层数 L
            layers[layer].append(item)
    top_words = [
        [re.sub(r'^L-\d+_K-\d+\s', '', item) for item in layer]
        for layer in layers.values()
    ]


    hierarchical_topic_dict = model_utils.convert_topicStr_to_dict(file_utils.read_text(f'{args.path}_T{T}'))
    if read_labels == 'True':
        train_labels = np.loadtxt(f'{dataset_path}/train_labels.txt', dtype=int)
        test_labels = np.loadtxt(f'{dataset_path}/test_labels.txt', dtype=int)

        # evaluate clustering
        results = clustering.hierarchical_clustering(test_theta, test_labels)
        print(dict(results))

        # evaluate classification
        results = classification.hierarchical_classification(train_theta, test_theta, train_labels, test_labels)
        print(dict(results))


    # compute topic coherence (C_V)
    TC = compute_topic_coherence(corpus, vocab, processed_top_words, cv_type=args.cv_type)
    print(f"C_V: {TC}")

    # compute topic diversity
    TD = multiaspect_topic_diversity(top_words)
    print(f"TD: {TD}")

    TU = multiaspect_topic_uniqueness(top_words)
    print(f"TU: {TU}")

    if run_hierarchy:
        phi_list = data_mat['phi_list']
        PC_pair_groups, PnonC_pair_groups, sibling_groups = get_topic_groups(hierarchical_topic_dict, beta_list, phi_list)

        # Parent and Child topic Coherence (PCC)
        CLNPMI = get_CLNPMI(PC_pair_groups, all_bow, vocab)
        print(f"===>CLNPMI_T{T}: {np.mean(CLNPMI):.5f}", "\t list: ", model_utils.round_list(CLNPMI))

        # Parent and Child topic Diversity (PCD)
        PC_TD = get_topics_difference(PC_pair_groups)
        print(f"===>PC_TD_T{T}: {np.mean(PC_TD):.5f}", "\t\t list: ", model_utils.round_list(PC_TD))

        # Sibling Topic Diversity (SD)
        Sibling_TD = get_Sibling_TD(sibling_groups)
        print(f"===>Sibling_TD_T{T}: {np.mean(Sibling_TD):.5f}", "\t list: ", model_utils.round_list(Sibling_TD))

        # Parent and non-Child Topic Diversity (PnCD)
        PnonC_TD = get_topics_difference(PnonC_pair_groups)
        print(f"===>PnonC_TD_T{T}: {np.mean(PnonC_TD):.5f}", "\t list: ", model_utils.round_list(PnonC_TD))
    else:
        print("===> hierarchy metrics skipped (no phi_list or --skip_hierarchy).")



if __name__ == '__main__':
    main()
