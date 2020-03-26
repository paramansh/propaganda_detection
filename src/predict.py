import os
import torch
import numpy as np
import argparse
import wget

import src.config
import src.input_processing
import src.pred_utils
import src.utils

from transformers import BertTokenizer

# parser = argparse.ArgumentParser()
# parser.add_argument('--interactive', nargs='?',  type=bool, default=False, help='Set True to enter custom input') 

# args = parser.parse_args()

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)

tokenizer = src.config.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(model_dir, 'model_370_44_bioe.pt')
if not os.path.exists(model_path):
  url = 'https://media.githubusercontent.com/media/paramansh/pd_models/master/model_370_44_bioe.pt'
  wget.download(url, model_path)

model = torch.load(model_path, map_location={'cuda:0':'cpu'})


def get_predictions(text):
  test_articles = [text]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = src.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = src.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  span_index_list = src.utils.return_spans(test_articles[0], sps[0])
  return span_index_list


if __name__ == "__main":
  # if args.interactive:
  #   print("Enter Input: ")
  #   input_sentence = input()
  #   test_articles = [input_sentence]
  # else:
  test_articles = ["Mini Mike, don’t lick your dirty fingers. Both unsanitary and dangerous to others and yourself!", "Just a random piece of text which should be a normal text"]
  print("Starting prediction")
  model = torch.load(os.path.join(model_dir, 'model_370_44_bioe.pt'), map_location={'cuda:0':'cpu'})
  test_spans = [[]] * len(test_articles)

  test_dataloader, test_sentences, test_bert_examples = src.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))

  sps = src.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  for i in range(len(test_articles)):
    print(test_articles[i])
    print('Detected span: ')
    src.utils.print_spans(test_articles[i], sps[i])
    print('--' * 50)
