# -*- coding: utf-8 -*-
'''
  @CreateTime	:  2022/05/18 12:27:25
  @Author	:  Alwin Zhang
  @Mail	:  zjfeng@homaytech.com
'''
import os
from sentence_transformers import SentenceTransformer, util
import torch

model_path = os.path.abspath('../../unsupervised_learning/SimCSE/output/training_simcse-uer/chinese_roberta_L-12_H-768-64-2022-05-18_12-42-48')
print(model_path)

embedder = SentenceTransformer(
    model_name_or_path=model_path)

# Corpus with example sentences
corpus = ['一个女人抱着一个男孩。',
          '一个男人在擦粉笔。',
          '那人在擦粉笔。',
          '那人在弹吉他。',
          '一个男人在弹吉他。',
          '一个男人在车库里举重。',
          '那人在骑马。',
          '一个女人在用锄头。'
          ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['一个人再擦粉笔', '一个人在弹吉他',
           '一个人在骑马']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
