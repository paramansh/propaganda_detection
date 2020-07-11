import os
import gdown
import torch

import classification
from classification import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaClassificationHead, CustomRobertaForSequenceClassification
test_articles = ["Just a random article 1", "Example article 2 with multiple spans"]
spans = [[(1, 4)], [(2, 7), (10, 14)]]
techniques = []
for span in spans:
  tech = []
  for sp in span:
    tech.append("Slogans")
  techniques.append(tech)
  
test_techniques, test_spans = techniques, spans 
test_dataloader = classification.get_data(test_articles, test_spans, test_techniques)

model_path = os.path.join(classification.model_dir, "classification_model_62.1_F1.pt")
if not os.path.exists(model_path):
  print('Downloading Model')
  url = 'https://drive.google.com/uc?id=13yFS3I4EBXI4QGFcAv8o1P32PwkkoT-S'
  gdown.download(url, model_path)

model = torch.load(model_path)
if classification.device == torch.device("cpu"):
  print("Using CPU for prediction")
  model = torch.load(model_path, map_location={'cuda:0':'cpu'})

pred, _ = classification.get_model_predictions(model, test_dataloader)
pred_i = 0
for index, span in enumerate(spans):
  article = test_articles[index]
  for sp in span:
    print(article[sp[0] : sp[1]], end = ": ")
    print(classification.distinct_techniques[pred[pred_i]])
    pred_i += 1
  print()


