from transformers import BertTokenizer

BATCH_SIZE = 8
TAGGING_SCHEME = "BIOE"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)
