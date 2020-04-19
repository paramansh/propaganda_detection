# from google.colab import files
import csv
import os
from classification import config
import classification
data_dir = config.data_dir

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

# Read training span labels 
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