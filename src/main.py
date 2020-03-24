import os
import sys
import torch
import csv
import numpy as np
import time
import datetime

# from utils import *
# from process import *

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm, trange
from shutil import copyfile

home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)

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

  for i, s in enumerate(start_positions):
    assert start_positions[i][0] == end_positions[i][0]
    if TAGGING_SCHEME == "BIO":
      current_propaganda_labels[start_positions[i][0]][start_positions[i][1]] = 2 # Begin label
      if start_positions[i][1] < end_positions[i][1]:
        current_propaganda_labels[start_positions[i][0]][start_positions[i][1] + 1 : end_positions[i][1] + 1] = [1] * (end_positions[i][1] - start_positions[i][1])
    if TAGGING_SCHEME == "BIOE":
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

def get_dataloader(examples, batch_size=8):
  inputs = torch.tensor([d.tokens_ids for d in examples])
  labels = torch.tensor([d.labels for d in examples])
  masks = torch.tensor([d.input_mask for d in examples])
  sentence_ids = torch.tensor([d.sentence_id for d in examples])
  tensor_data = TensorDataset(inputs, labels, masks, sentence_ids)
  dataloader = DataLoader(tensor_data, batch_size=BATCH_SIZE)
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
    input_feature = convert_sentence_to_input_feature(sentence, i, tokenizer)
    bert_examples.append(input_feature)
  dataloader = get_dataloader(bert_examples, BATCH_SIZE)
  return dataloader, sentences, bert_examples

def train(model, train_dataloader, eval_dataloader, epochs=5, save_model=False):
  max_grad_norm = 1.0

  for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
      # add batch to gpu
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_ids = batch
      loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      loss.backward()
      tr_loss += loss.item()
      nb_tr_examples += b_input_ids.size(0)
      nb_tr_steps += 1
      torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
      optimizer.step()
      model.zero_grad()
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    get_score(model, mode="train")

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in eval_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_ids = batch
      with torch.no_grad():
        tmp_eval_loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)     
      eval_loss += tmp_eval_loss.mean().item()
      # eval_accuracy += tmp_eval_accuracy
      
      nb_eval_examples += b_input_ids.size(0)
      nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))

    get_score(model, mode="eval")
    if save_model:
      model_name = 'model_' + str(datetime.datetime.now()) + '.pt'
      torch.save(model, os.path.join(model_dir, model_name))
      print("Model saved:", model_name)
    print()
    time.sleep(1)



def read_articles(article_dir):
  articles = []
  train_dir = os.path.join(data_dir, article_dir)
  for filename in sorted(os.listdir(train_dir)):
    myfile = open(os.path.join(train_dir, filename))
    article = myfile.read()
    articles.append(article)
    myfile.close()
  article_ids = []
  for filename in sorted(os.listdir(train_dir)):
    article_ids.append(filename[7:-4])
  return articles, article_ids

def read_spans():
  spans = []
  label_dir = os.path.join(data_dir, "train-labels-task1-span-identification")
  for filename in sorted(os.listdir(label_dir)):
    myfile = open(os.path.join(label_dir, filename))
    tsvreader = csv.reader(myfile, delimiter="\t")
    span = []
    for row in tsvreader:
      span.append((int(row[1]), int(row[2])))
    myfile.close()
    spans.append(span)
  return spans

def print_spans(article, span):
  for sp in span:
    print (article[sp[0]: sp[1]])
  print()

def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=2).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_model_predictions(model, dataloader):
  model.eval()
  predictions , true_labels, sentence_ids = [], [], []
  nb_eval_steps = 0
  for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_labels, b_input_mask, b_ids = batch  
    with torch.no_grad():
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = logits[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    s_ids = b_ids.to('cpu').numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    # print(label_ids)
    true_labels.extend(label_ids)
    sentence_ids.extend(s_ids)
    nb_eval_steps += 1
  
  return predictions, true_labels, sentence_ids

def merge_spans(current_spans):
  if not current_spans:
    return [] 
  merged_spans = []
  li = current_spans[0][0]
  ri = current_spans[0][1]
  threshold = 2
  for i in range(len(current_spans) - 1):
    span = current_spans[i+1]
    if span[0] - ri < 2:
      ri = span[1]
      continue
    else:
      merged_spans.append((li, ri))
      li = span[0]
      ri = span[1]
  merged_spans.append((li, ri))
  return merged_spans

def get_score(model, mode=None):
  predicted_spans = [[] for i in range(400)] # TODO 400 hardcoded
  
  def get_span_prediction(prediction_labels, sentence_index, sentences, bert_examples):
    index = sentence_index 
    bert_example = bert_examples[index]
    mask = bert_example.input_mask
    pred_labels_masked = prediction_labels # need to change to predictions later
    pred_labels = []
    for i, m in enumerate(mask):
      if m > 0:
        pred_labels.append(pred_labels_masked[i])
    if bert_example.add_cls_sep:
      pred_labels.pop() # remove ['SEP'] label
      pred_labels.pop(0) # remove ['CLS'] label

    sentence = sentences[index]
    sent_len = len(sentence.tokens)
    final_pred_labels = [0] * sent_len
    cur_map = bert_example.tok_to_orig_index
    for i, label in enumerate(pred_labels):
      final_pred_labels[cur_map[i]] |= label
    # assert final_pred_labels == sentence.labels
    
    word_start_index_map = sentence.word_to_start_char_offset
    word_end_index_map = sentence.word_to_end_char_offset

    article_index = sentence.article_index
    for i, label in enumerate(final_pred_labels):
      if label:
        # print(word_start_index_map[i], word_end_index_map[i])
        predicted_spans[article_index].append((word_start_index_map[i], word_end_index_map[i]))
  
  if mode == "train":
    indices = train_indices
    predictions, true_labels, sentence_ids = get_model_predictions(model, train_dataloader)
    pred_sentences, pred_bert_examples = train_sentences, train_bert_examples
  elif mode == "test":
    predictions, true_labels , sentence_ids = get_model_predictions(model, test_dataloader)
    pred_sentences, pred_bert_examples = test_sentences, test_bert_examples
  else:
    indices = eval_indices
    predictions, true_labels, sentence_ids = get_model_predictions(model, eval_dataloader)
    pred_sentences, pred_bert_examples = eval_sentences, eval_bert_examples

  merged_predicted_spans = []
  # TODO sorting of spans???? may not be in order??
  for ii, _ in enumerate(predictions):
    get_span_prediction(predictions[ii], sentence_ids[ii], pred_sentences, pred_bert_examples)
  for span in predicted_spans:
    merged_predicted_spans.append(merge_spans(span))
  if mode == "test":
    return merged_predicted_spans 
  if not os.path.isdir("predictions"):
    os.mkdir("predictions")
  copyfile("gdrive/My Drive/propaganda_detection/tools/task-SI_scorer.py", "predictions/task-SI_scorer.py")
  with open("predictions/predictions.tsv", 'w') as fp:
    for index in indices:
      filename = "article" + article_ids[index] + ".task1-SI.labels"
      copyfile(os.path.join(data_dir, "train-labels-task1-span-identification/" + filename), "predictions/" + filename)
      for ii in merged_predicted_spans[index]:
        fp.write(article_ids[index] + "\t" + str(ii[0]) + "\t" + str(ii[1]) + "\n")

  # !python3 predictions/task-SI_scorer.py -s predictions/predictions.tsv -r predictions/ -m

  for index in indices:
    filename = "article" + article_ids[index] + ".task1-SI.labels"
    os.remove("predictions/" + filename)




# articles, article_ids = read_articles('train-articles')
# spans = read_spans()
# TAGGING_SCHEME = "PN" # Positive Negative
TAGGING_SCHEME = "BIOE"
# NUM_ARTICLES = len(articles)
# NUM_ARTICLES = 100
# articles = articles[0:NUM_ARTICLES]
# spans = spans[0:NUM_ARTICLES]
BATCH_SIZE=8
# np.random.seed(245)
# indices = np.arange(NUM_ARTICLES)
# np.random.shuffle(indices)
# train_indices = indices[:int(0.9 * NUM_ARTICLES)]
# eval_indices = indices[int(0.9 * NUM_ARTICLES):]

# bert_model_class = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)

# train_dataloader, train_sentences, train_bert_examples = get_data(articles, spans, train_indices)
# eval_dataloader, eval_sentences, eval_bert_examples = get_data(articles, spans, eval_indices)

# num_labels = 2 + int(TAGGING_SCHEME == "BIO") + 2 * int(TAGGING_SCHEME == "BIOE")
# model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# model.cuda()
# if TAGGING_SCHEME == "BIOE":
#   WEIGHTS = torch.tensor([1.0, 5.0, 10.0, 5.0]).cuda()
# else:
#   WEIGHTS = torch.tensor([1.0, 100.0]).cuda()


# from torch.optim import Adam
# FULL_FINETUNING = True 
# if FULL_FINETUNING:
#   param_optimizer = list(model.named_parameters())
#   no_decay = ['bias', 'gamma', 'beta']
#   optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#       'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#       'weight_decay_rate': 0.0}
#   ]
# else:
#   param_optimizer = list(model.classifier.named_parameters()) 
#   optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
# optimizer = Adam(optimizer_grouped_parameters, lr=3e-5) # lr 3e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train(model, train_dataloader, eval_dataloader, epochs=4, save_model=(NUM_ARTICLES >= 150))

print("Starting prediction")
test_articles = ["Mini Mike, don’t lick your dirty fingers. Both unsanitary and dangerous to others and yourself!", "Just a random piece of text which should be a normal text"]
ind = 0
model = torch.load(os.path.join(model_dir, 'model_370_44_bioe.pt'), map_location={'cuda:0':'cpu'})
# model = torch.load(os.path.join(model_dir, 'model_370_43_bio.pt'), map_location={'cuda:0':'cpu'})

# test_articles = [articles[ind]]
test_spans = [[]] * len(test_articles)

test_dataloader, test_sentences, test_bert_examples = get_data(test_articles, test_spans, indices=np.arange(len(test_articles)))
sps = get_score(model, mode="test")
for i in range(len(test_articles)):
  print(test_articles[i])
  print('Detected span: ')
  print_spans(test_articles[i], sps[i])
  print('--' * 50)
# print_spans(articles[ind], spans[ind])
