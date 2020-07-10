import os
import torch
import numpy as np
import argparse
import wget
import gdown

from src import identification

from transformers import BertTokenizer

# parser = argparse.ArgumentParser()
# parser.add_argument('--interactive', nargs='?',  type=bool, default=False, help='Set True to enter custom input') 

# args = parser.parse_args()

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)

tokenizer = identification.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(model_dir, 'model_370_44_bioe.pt')
if not os.path.exists(model_path):
  url = 'https://drive.google.com/uc?id=1-5oN2lS37IcXT1Lhd-H3TxlEdi4MzVPC'
  gdown.download(url, model_path)

model = torch.load(model_path, map_location={'cuda:0':'cpu'})

def get_dev_outputs(article_dir="dev-articles"):
  test_articles, test_article_ids = identification.read_articles('dev-articles')
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.get_score(model,
      dataloader=test_dataloader,
      sentences=test_sentences,
      bert_examples=test_bert_examples,
      mode="test")
  with open('dev_predictions.txt', 'w') as fp:
    for index in range(len(test_articles)):
      for ii in sps[index]:
        fp.write(test_article_ids[index] + "\t" + str(ii[0]) + "\t" + str(ii[1]) + "\n")

def get_predictions(text):
  test_articles = [text]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  spans = identification.utils.return_spans(test_articles[0], sps[0])
  return spans

def get_predictions_indices(text):
  test_articles = [text]
  test_spans = [[]] * len(test_articles)
  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  return sps[0]



if __name__ == "__main":
  # if args.interactive:
  #   print("Enter Input: ")
  #   input_sentence = input()
  #   test_articles = [input_sentence]
  # else:
  test_articles = ["A propaganda jihadi test to be done!", "Just a random piece of text which should be a normal text"]
  print("Starting prediction")
  model = torch.load(os.path.join(model_dir, 'model_370_44_bioe.pt'), map_location={'cuda:0':'cpu'})
  test_spans = [[]] * len(test_articles)

  test_dataloader, test_sentences, test_bert_examples = identification.input_processing.get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))

  sps = identification.pred_utils.get_score(model, 
                            dataloader=test_dataloader,
                            sentences=test_sentences,
                            bert_examples=test_bert_examples,
                            mode="test")

  for i in range(len(test_articles)):
    print(test_articles[i])
    print('Detected span: ')
    identification.utils.print_spans(test_articles[i], sps[i])
    print('--' * 50)
