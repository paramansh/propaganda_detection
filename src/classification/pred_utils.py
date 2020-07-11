# from google.colab import files
import csv
import os
import torch
import numpy as np

from . import config
import classification

device = config.device
data_dir = config.data_dir

def get_model_predictions(model, dataloader):
  model.eval()
  predictions , true_labels = [], []
  nb_eval_steps = 0
  for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_labels, b_input_mask, b_lengths = batch
    with torch.no_grad():
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, lengths=b_lengths)
    logits = logits[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    pred_label = np.argmax(logits, axis=1)
    predictions.extend(pred_label)
    true_labels.extend(label_ids)
  return predictions, true_labels

def get_dev_predictions(model):
  test_articles, _ = classification.read_articles("dev-articles")
  test_spans, test_techniques = classification.read_test_spans()

  test_articles = test_articles[1:]
  test_dataloader = classification.get_data(test_articles, test_spans, test_techniques)
  pred, _ = classification.get_model_predictions(model, test_dataloader)

  with open('predictions.txt', 'w') as fp:
    label_file = os.path.join(data_dir, "dev-task-TC-template.out")
    myfile = open(label_file)
    tsvreader = csv.reader(myfile, delimiter="\t")
    for i, row in enumerate(tsvreader):
      fp.write(row[0] + '\t' + config.distinct_techniques[pred[i]] + '\t' + row[2] + '\t' + row[3] + '\n')
  # files.download('predictions.txt')

def get_test_predictions(model):
  temp_test_articles, test_indices = classification.read_articles("test-TC/test-articles")
  test_spans, test_techniques, span_indices = classification.read_test_spans(mode="test")
  test_articles = []
  span_indices = set(span_indices)
  for index, article in enumerate(temp_test_articles):
    if test_indices[index] in span_indices:
      test_articles.append(article)
  # test_articles = test_articles[1:]
  print(len(test_articles))
  print(len(test_spans))
  test_dataloader = classification.get_data(test_articles, test_spans, test_techniques)
  pred, _ = classification.get_model_predictions(model, test_dataloader)

  with open('predictions.txt', 'w') as fp:
    label_file = os.path.join(data_dir, "test-TC/test-task-TC-template.out")
    myfile = open(label_file)
    tsvreader = csv.reader(myfile, delimiter="\t")
    for i, row in enumerate(tsvreader):
      fp.write(row[0] + '\t' + config.distinct_techniques[pred[i]] + '\t' + row[2] + '\t' + row[3] + '\n')
  # files.download('predictions.txt')
