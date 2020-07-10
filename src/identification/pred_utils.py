import os
import torch
import numpy as np
from shutil import copyfile
from . import config

def merge_spans(current_spans):
  if not current_spans:
    return [] 
  merged_spans = []
  li = current_spans[0][0]
  ri = current_spans[0][1]
  threshold = 2
  for i in range(len(current_spans) - 1):
    span = current_spans[i+1]
    if span[0] - ri < threshold:
      ri = span[1]
      continue
    else:
      merged_spans.append((li, ri))
      li = span[0]
      ri = span[1]
  merged_spans.append((li, ri))
  return merged_spans


def get_model_predictions(model, dataloader):
  model.eval()
  predictions , true_labels, sentence_ids = [], [], []
  nb_eval_steps = 0
  for batch in dataloader:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_labels, b_input_mask, b_ids = batch  
    with torch.no_grad():
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = logits[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    s_ids = b_ids.to('cpu').numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(label_ids)
    sentence_ids.extend(s_ids)
    nb_eval_steps += 1
  return predictions, true_labels, sentence_ids


def get_score(model, dataloader, sentences, bert_examples, mode=None, article_ids=None, indices=None):
  predicted_spans = [[] for i in range(400)] # TODO 400 hardcoded
  
  def get_span_prediction(prediction_labels, sentence_index, sentences, bert_examples):
    index = sentence_index 
    bert_example = bert_examples[index]
    mask = bert_example.input_mask
    pred_labels_masked = prediction_labels 
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
        predicted_spans[article_index].append((word_start_index_map[i], word_end_index_map[i]))
  
  predictions, _, sentence_ids = get_model_predictions(model, dataloader)
  pred_sentences, pred_bert_examples = sentences, bert_examples

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
  copyfile(os.path.join(config.home_dir, "tools/task-SI_scorer.py"), "predictions/task-SI_scorer.py")
  with open("predictions/predictions.tsv", 'w') as fp:
    for index in indices:
      filename = "article" + article_ids[index] + ".task1-SI.labels"
      copyfile(os.path.join(config.data_dir, "train-labels-task1-span-identification/" + filename), "predictions/" + filename)
      for ii in merged_predicted_spans[index]:
        fp.write(article_ids[index] + "\t" + str(ii[0]) + "\t" + str(ii[1]) + "\n")

  os.system("python3 predictions/task-SI_scorer.py -s predictions/predictions.tsv -r predictions/ -m")

  for index in indices:
    filename = "article" + article_ids[index] + ".task1-SI.labels"
    os.remove("predictions/" + filename)
