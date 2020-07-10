# from transformers import BertTokenizer
from transformers import AlbertTokenizer
import os

BATCH_SIZE = 8
NUM_ARTICLES = 10
TAGGING_SCHEME = "BIOE"
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', lower_case=True)

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
