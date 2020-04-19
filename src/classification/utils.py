import os
import csv
import classification.config as config
import datetime
import numpy as np

from sklearn import metrics

data_dir = config.data_dir

def read_articles(dir_name):
  articles = []
  train_dir = os.path.join(data_dir, dir_name)
  for filename in sorted(os.listdir(train_dir)):
    myfile = open(os.path.join(train_dir, filename))
    article = myfile.read()
    articles.append(article)
    myfile.close()
  article_ids = []
  for filename in sorted(os.listdir(train_dir)):
    article_ids.append(int(filename[7:-4]))
  return articles, article_ids

def read_spans(mode=None):
  spans = []
  techniques = []
  if mode == "test":
    label_dir = os.path.join(data_dir, "dev-task-TC-template.out")
  else:
    label_dir = os.path.join(data_dir, "train-labels-task2-technique-classification")
  for filename in sorted(os.listdir(label_dir)):
    myfile = open(os.path.join(label_dir, filename))
    tsvreader = csv.reader(myfile, delimiter="\t")
    span = []
    technique = []
    for row in tsvreader:
      span.append((int(row[2]), int(row[3])))
      if mode == "test":
        technique.append("Slogans") # DUMMY
      else:
        technique.append(row[1])
    myfile.close()
    spans.append(span)
    techniques.append(technique)
  return spans, techniques

def read_test_spans(mode=None):
  spans = []
  techniques = []
  indices = []
  if mode == 'test':
    label_file = os.path.join(data_dir, "test-TC/test-task-TC-template.out")
  else:  
    label_file = os.path.join(data_dir, "dev-task-TC-template.out")
  myfile = open(label_file)
  prev_index = -1
  tsvreader = csv.reader(myfile, delimiter="\t")

  span = []
  technique = []
  for row in tsvreader:
    article_index = int(row[0])
    if article_index != prev_index:
      if prev_index != -1:
        spans.append(span)
        techniques.append(technique)
      span = []
      technique = []
      span.append((int(row[2]), int(row[3])))
      technique.append("Slogans")
      indices.append(article_index)
      prev_index = article_index
    else:
      span.append((int(row[2]), int(row[3])))
      technique.append("Slogans")
  spans.append(span)
  techniques.append(technique)
  indices.append(article_index)
  if mode == 'test':
    return spans, techniques, indices
  return spans, techniques

def print_spans(article, span, technique):
  for index, sp in enumerate(span):
    print(technique[index], config.tag2idx[technique[index]], end=' - ')
    print (article[sp[0]: sp[1]])
  print()

def compute_metrics(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  print(metrics.confusion_matrix(labels_flat, pred_flat))
  print(metrics.classification_report(labels_flat, pred_flat))

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
  elapsed_rounded = int(round((elapsed)))
  return str(datetime.timedelta(seconds=elapsed_rounded))

