import os
import torch
import numpy as np
import argparse

import config
import input_processing
import pred_utils
import utils

from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', nargs='?',  type=bool, default=False, help='Set True to enter custom input') 

args = parser.parse_args()

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)

tokenizer = config.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.interactive:
  print("Enter Input: ")
  input_sentence = input()
  test_articles = [input_sentence]
else:
  test_articles = ["Mini Mike, don’t lick your dirty fingers. Both unsanitary and dangerous to others and yourself!", "Just a random piece of text which should be a normal text"]
print("Starting prediction")
model = torch.load(os.path.join(model_dir, 'model_370_44_bioe.pt'), map_location={'cuda:0':'cpu'})
test_spans = [[]] * len(test_articles)

test_dataloader, test_sentences, test_bert_examples = input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))

sps = pred_utils.get_score(model, 
                           dataloader=test_dataloader,
                           sentences=test_sentences,
                           bert_examples=test_bert_examples,
                           mode="test")

for i in range(len(test_articles)):
  print(test_articles[i])
  print('Detected span: ')
  utils.print_spans(test_articles[i], sps[i])
  print('--' * 50)
