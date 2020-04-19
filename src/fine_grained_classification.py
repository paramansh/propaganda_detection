import os
import sys
import torch
import csv
import numpy as np
import random
import time
import datetime
import pprint

import classification

from sklearn.model_selection import train_test_split

if not os.path.isdir(classification.model_dir):
  os.mkdir(classification.model_dir)

print('reached reading articles')
articles, article_ids = classification.read_articles("train-articles")
spans, techniques = classification.read_spans()
pprint.pprint(classification.tag2idx)

NUM_ARTICLES = len(articles)
NUM_ARTICLES = 10
articles = articles[0:NUM_ARTICLES]
spans = spans[0:NUM_ARTICLES]
techniques = techniques[0:NUM_ARTICLES]

print(len(spans))
print(classification.BATCH_SIZE)
print(classification.device)

# seed_val = 1328 # 32
seed_val = 32 # 32
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

indices = np.arange(NUM_ARTICLES)

train_articles, eval_articles, train_spans, eval_spans, train_techniques, eval_techniques, train_indices, eval_indices = train_test_split(articles, spans, techniques, indices, test_size=0.2)

train_dataloader = classification.get_data(train_articles, train_spans, train_techniques)
eval_dataloader = classification.get_data(eval_articles, eval_spans, eval_techniques)

model = classification.pretrained_model
if torch.cuda.is_available():
  print('Using GPU')
  model.cuda()

classification.train(model, train_dataloader, eval_dataloader, epochs=2)
