import os

BATCH_SIZE = 8
NUM_ARTICLES = 10
TAGGING_SCHEME = "BIOE"
LANGUAGE_MODEL = "BERT" # default
if LANGUAGE_MODEL == "Albert":
  from transformers import AlbertTokenizer
  tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', lower_case=True)
else:
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")

model_dir = os.path.join(home_dir, "model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
