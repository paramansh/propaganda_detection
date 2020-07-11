from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from . import config
import torch

def get_examples(articles, spans, techniques):
  assert len(articles) == len(spans) and len(spans) == len(techniques)
  sentences = []
  labels = []
  for index, article in enumerate(articles):
    span = spans[index]
    technique = techniques[index]
    assert len(technique) == len(span)
    for i, sp in enumerate(span):
      pt = config.tag2idx[technique[i]]
      sentence = article[sp[0]: sp[1]]
      sentences.append(sentence)
      labels.append(pt)
  return sentences, labels

def convert_sentence_to_input_feature(sentence, tokenizer, add_cls_sep=True, max_seq_len=150):
  tokenized_sentence = tokenizer.encode_plus(sentence,
                                             add_special_tokens=add_cls_sep,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True)
  return tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']

def get_data(articles, spans, techniques):
  sentences, labels = get_examples(articles, spans, techniques)
  attention_masks = []
  inputs = []
  lengths = []
  for i, sentence in enumerate(sentences):
    lengths.append(len(sentence) / 100) # divide by 100 for normalization
    input_ids, mask = convert_sentence_to_input_feature(sentence, config.tokenizer)
    inputs.append(input_ids)
    attention_masks.append(mask)
  
  inputs = torch.tensor(inputs)
  labels = torch.tensor(labels)
  masks = torch.tensor(attention_masks)
  lengths = torch.tensor(lengths).float()
  tensor_data = TensorDataset(inputs, labels, masks, lengths)
  dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE)
  return dataloader

