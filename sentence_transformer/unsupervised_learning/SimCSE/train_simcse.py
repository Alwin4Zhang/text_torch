# -*- coding: utf-8 -*-
'''
  @CreateTime	:  2022/05/18 10:40:39
  @Author	:  Alwin Zhang
  @Mail	:  zjfeng@homaytech.com
'''

import math
import os
import json
import gzip
import csv
import logging
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import BertTokenizer, BertModel, BertConfig

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

train_batch_size = 64
max_seq_length = 64
num_epochs = 3

model_name = "uer/chinese_roberta_L-12_H-768"

model_save_path = './output/training_simcse-{}-{}'.format(model_name, train_batch_size)


sts_file_path = "/home/homay/zjfeng/sen_transformers/data/senteval_cn/STS-B/"
sts_train_file = "STS-B.train.data"
sts_test_file = "STS-B.test.data"
sts_dev_file = "STS-B.valid.data"

# snli_file_path = "./cnsd-snli/"
# snli_train_file = 'cnsd_snli_v1.0.trainproceed.txt'

# def load_snli_vocab(path):
#     data = []
#     with open(path) as f:
#         for i in f:
#             data.append(json.loads(i)['origin'])
#     return data


def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            i = i.strip()
            d = i.split("\t")
            sentence1 = d[0]
            sentence2 = d[1]
            score = float(d[2]) / 5.0
            data.append(InputExample(
                texts=[sentence1, sentence2], label=score))
    return data


word_embedding_model = models.Transformer(
    model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_samples = load_STS_data(os.path.join(sts_file_path, sts_train_file))
test_samples = load_STS_data(os.path.join(sts_file_path, sts_test_file))
dev_samples = load_STS_data(os.path.join(sts_file_path, sts_dev_file))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, batch_size=train_batch_size, name="sts-b-dev")
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, batch_size=train_batch_size, name="sts-b-test")

train_dataloader = DataLoader(
    train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

train_loss = losses.MultipleNegativesRankingLoss(model)

# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
# Evaluate every 10% of the data
evaluation_steps = int(len(train_dataloader) * 0.1)
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 5e-5},
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)