import torch

from . import config

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class example_sentence:
  def __init__(self):
    self.tokens = []
    self.labels = []
    self.article_index = -1 # index of the article to which the sentence is associated
    self.index = -1 # index of the sentence in that article 
    self.word_to_start_char_offset = []
    self.word_to_end_char_offset = []
  
  def __str__(self):
    print("tokens -", self.tokens)
    print("labels -", self.labels)
    print("article_index -", self.article_index)
    print("index -", self.index)
    print("start_offset -", self.word_to_start_char_offset)
    print("end_offset -", self.word_to_end_char_offset)   
    return "" 

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False

def get_sentence_tokens_labels(article, span=None, article_index=None):
  doc_tokens = []
  char_to_word_offset = []
  current_sentence_tokens = [] # actually all sentence tokens for particular article. #TODO rename
  word_to_start_char_offset = {}
  word_to_end_char_offset = {}
  prev_is_whitespace = True
  prev_is_newline = True
  current_word_position = None
  for index, c in enumerate(article):
    if c == "\n":
      prev_is_newline = True
      # check for empty lists
      if doc_tokens:
        current_sentence_tokens.append(doc_tokens)
      doc_tokens = []
    if is_whitespace(c):
      prev_is_whitespace = True
      if current_word_position is not None:
        word_to_end_char_offset[current_word_position] = index
        current_word_position = None
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
        current_word_position = (len(current_sentence_tokens), len(doc_tokens) - 1)
        word_to_start_char_offset[current_word_position] = index # start offset of word
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append((len(current_sentence_tokens), len(doc_tokens) - 1))
  if doc_tokens:
    current_sentence_tokens.append(doc_tokens)
  if current_word_position is not None:
    word_to_end_char_offset[current_word_position] = index
    current_word_position = None
  if span is None:
    return current_sentence_tokens, (word_to_start_char_offset, word_to_end_char_offset)

  current_propaganda_labels = []
  for doc_tokens in current_sentence_tokens:
    current_propaganda_labels.append([0] * len(doc_tokens))

  start_positions = []
  end_positions = []

  for sp in span:
    if (char_to_word_offset[sp[0]][0] != char_to_word_offset[sp[1]-1][0]):
      l1 = char_to_word_offset[sp[0]][0]
      l2 = char_to_word_offset[sp[1] - 1][0]
      start_positions.append(char_to_word_offset[sp[0]])
      end_positions.append((l1, len(current_sentence_tokens[l1])-1))
      l1 += 1
      while(l1 < l2):
        start_positions.append((l1, 0))
        end_positions.append((l1, len(current_sentence_tokens[l1])-1))
        l1 += 1
      start_positions.append((l2, 0))
      end_positions.append(char_to_word_offset[sp[1]-1])  
      continue
    start_positions.append(char_to_word_offset[sp[0]])
    end_positions.append(char_to_word_offset[sp[1]-1])

  for i, _ in enumerate(start_positions):
    assert start_positions[i][0] == end_positions[i][0]
    if config.TAGGING_SCHEME == "BIO":
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
      if start_positions[i][1] < end_positions[i][1]:
        current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1 : end_positions[i][1] + 1] = [1] * (end_positions[i][1] - start_positions[i][1])
    if config.TAGGING_SCHEME == "BIOE":
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
      if start_positions[i][1] < end_positions[i][1]:
        current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1 : end_positions[i][1]] = [1] * (end_positions[i][1] - start_positions[i][1] - 1)
        current_propaganda_labels[start_positions[i][0]][end_positions[i][1]] = 3 # End label
    else:
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1] : end_positions[i][1] + 1] = [1] * (end_positions[i][1] + 1 - start_positions[i][1])
  
  num_sentences = len(current_sentence_tokens)

  start_offset_list = get_list_from_dict(num_sentences, word_to_start_char_offset)
  end_offset_list = get_list_from_dict(num_sentences, word_to_end_char_offset)
  sentences = []
  for i in range(num_sentences):
    sentence = example_sentence()
    sentence.tokens = current_sentence_tokens[i]
    sentence.labels = current_propaganda_labels[i]
    sentence.article_index =  article_index
    sentence.index = i
    sentence.word_to_start_char_offset = start_offset_list[i]
    sentence.word_to_end_char_offset = end_offset_list[i]
    num_words = len(sentence.tokens)
    assert len(sentence.labels) == num_words
    assert len(sentence.word_to_start_char_offset) == num_words
    assert len(sentence.word_to_end_char_offset) == num_words
    sentences.append(sentence)

  return current_sentence_tokens, current_propaganda_labels, (word_to_start_char_offset, word_to_end_char_offset), sentences

def get_list_from_dict(num_sentences, word_offsets):
  li = []
  for _ in range(num_sentences):
    li.append([])
  for key in word_offsets:
    si = key[0]
    li[si].append(word_offsets[key])

  return li

class BertExample:
  def __init__(self):
    self.add_cls_sep = True
    self.sentence_id = -1
    self.orig_to_tok_index = []
    self.tok_to_orig_index = []
    self.labels = None
    self.tokens_ids = []
    self.input_mask = []
  def __str__(self):
    print("sentence_id", self.sentence_id)
    return ""

def convert_sentence_to_input_feature(sentence, sentence_id, tokenizer, add_cls_sep=True, max_seq_len=256):
  bert_example = BertExample()
  bert_example.sentence_id = sentence_id
  bert_example.add_cls_sep = add_cls_sep

  sentence_tokens = sentence.tokens
  sentence_labels = sentence.labels 

  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = [] 
  for (i, token) in enumerate(sentence_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)
  bert_example.tok_to_orig_index = tok_to_orig_index
  bert_example.orig_to_tok_index = orig_to_tok_index

  bert_tokens = all_doc_tokens
  if add_cls_sep:
    bert_tokens = ["[CLS]"] + bert_tokens
    bert_tokens = bert_tokens + ["[SEP]"]
  
  tokens_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
  input_mask = [1] * len(tokens_ids)
  while len(tokens_ids) < max_seq_len:
    tokens_ids.append(0)
    input_mask.append(0)
  # tokens_ids = pad_sequences(tokens_ids, maxlen=max_seq_len, truncating="post", padding="post", dtype="int")
  bert_example.tokens_ids = tokens_ids
  bert_example.input_mask = input_mask
  # bert_example.input_mask = [float(i>0) for i in token_ids]

  if sentence_labels is None:
    return bert_example
  

  labels = [0] * len(all_doc_tokens)
  for index, token in enumerate(all_doc_tokens):
    labels[index] = sentence_labels[tok_to_orig_index[index]]
  if add_cls_sep:
    labels = [0] + labels
    labels = labels + [0]
  # labels = pad_sequences(labels, maxlen=max_seq_len, truncating="post", padding="post", dtype="int")
  while len(labels) < max_seq_len:
    labels.append(0)
  bert_example.labels = labels

  return bert_example 

def get_dataloader(examples):
  inputs = torch.tensor([d.tokens_ids for d in examples])
  labels = torch.tensor([d.labels for d in examples])
  masks = torch.tensor([d.input_mask for d in examples])
  sentence_ids = torch.tensor([d.sentence_id for d in examples])
  tensor_data = TensorDataset(inputs, labels, masks, sentence_ids)
  dataloader = DataLoader(tensor_data, batch_size=config.BATCH_SIZE)
  return dataloader

def get_data(articles, spans, indices):
  assert len(articles) == len(spans)    
  sentences = []
  for index in indices:
    article = articles[index]
    span = spans[index]
    _, _, _, cur_sentences = get_sentence_tokens_labels(article, span, index)
    sentences += cur_sentences

  print(len(sentences))
  print(max([len(s.tokens) for s in sentences]))

  bert_examples = []
  for i, sentence in enumerate(sentences):
    input_feature = convert_sentence_to_input_feature(sentence, i, config.tokenizer)
    bert_examples.append(input_feature)
  dataloader = get_dataloader(bert_examples)
  return dataloader, sentences, bert_examples

