import os
import topmost
import json
import numpy as np
import time
import gc
import torch
from topmost import evaluations
from topmost.data import file_utils




# 你的训练和评估逻辑
dataset_name_list = ["20NG","ACL"]
# dataset_name_list = ["IMDB"]
for dataset_name in dataset_name_list:
    for index in range(4,7):
        # device = "cpu"  or "cuda"++
        device = "cuda"
        # dataset_name = "ACL"
        dataset_dir = f"./data/{dataset_name}"

        model_name = "ECRTM"
        start_time = time.time()
        if dataset_name in ["NYT", "20NG", "IMDB"]:
            dataset = topmost.data.BasicDataset(dataset_dir, read_labels=True, device=device)
        else:
            dataset = topmost.data.BasicDataset(dataset_dir, read_labels=False, device=device)


        model = topmost.models.ECRTM(dataset.vocab_size, num_topics=50)
        model = model.to(device)
        trainer = topmost.trainers.BasicTrainer(model, dataset, verbose=True)
        top_words, train_theta = trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"模型训练时间为{total_time}")
        # 保存embeddings
        # topic_embeddings_list_np = [emb.detach().cpu().numpy() for emb in model.topic_embeddings_list]
        # word_embeddings_np = model.bottom_word_embeddings.detach().cpu().numpy()
        # np.savez_compressed(
        #     f'output/{dataset_name}/{model_name}_embeddings.npz',
        #     topic_embeddings_list=np.array(topic_embeddings_list_np, dtype=object),
        #     word_embeddings=word_embeddings_np
        # )

        # 保存topwords
        # topic_str_list = []
        # for layer, level_top_words in enumerate(top_words):
        #     for topic_idx, words in enumerate(level_top_words):
        #         topic_str_list.append(f"L-{layer}_K-{topic_idx} {words}")

        topic_str_list = []
        for topic_idx, words in enumerate(top_words):
            topic_str_list.append(f"K-{topic_idx} {words}")

        output_prefix = f'output/{dataset_name}/{model_name}_K50_{index}th'
        file_utils.make_dir(os.path.dirname(output_prefix))
        file_utils.save_text(topic_str_list, f'{output_prefix}_T15')


        output_eva = f'output/{dataset_name}_eva/{model_name}_K50_{index}th'
        file_utils.make_dir(os.path.dirname(output_eva))


        # get theta (doc-topic distributions)
        train_theta, test_theta = trainer.export_theta()

        # compute topic coherence
        corpus = trainer.dataset.train_texts
        vocab = trainer.dataset.vocab
        # flattened_top_words = [topic for layer in top_words for topic in layer]
        # print(top_words)

        TC = evaluations.compute_topic_coherence(corpus, vocab, top_words)
        print(f"TC: {TC}")
        file_utils.save_text([f"TC: {TC}"], f'{output_eva}_TC')


        # compute topic diversity
        TD = evaluations.compute_topic_diversity(top_words)
        print(f"TD: {TD}")
        file_utils.save_text([f"TD: {TD}"], f'{output_eva}_TD')
        del model
        del trainer
        del dataset
        del train_theta
        del test_theta
        del top_words

        gc.collect()
        torch.cuda.empty_cache()

        # if dataset_name in ["NYT","20NG"]:
        #     # evaluate clustering
        #     results = evaluations.hierarchical_clustering(test_theta, dataset.test_labels)
        #     print(dict(results))
        #     file_utils.save_text(json.dumps(results, indent=4).splitlines(), f'{output_eva}_clustering')
        #
        #     # evaluate classification
        #     results = evaluations.hierarchical_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
        #     print(dict(results))
        #     file_utils.save_text(json.dumps(results, indent=4).splitlines(), f'{output_eva}_classification')

        # evaluate quality of topic hierarchy
        # beta_list = trainer.get_beta()
        # phi_list = trainer.get_phi()
        # annoated_top_words = trainer.get_top_words(annotation=True)
        # reference_bow = np.concatenate((dataset.train_bow, dataset.test_bow), axis=0) # or reference_bow = train_bow
        # results, topic_hierarchy = evaluations.hierarchy_quality(dataset.vocab, reference_bow, annoated_top_words, beta_list, phi_list)
        # file_utils.save_text(json.dumps(results, indent=4).splitlines(), f'{output_eva}_quality of topic hierarchy')


        # print(json.dumps(topic_hierarchy, indent=4))
        #     print(results)


