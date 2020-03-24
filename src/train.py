import time
import datetime

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
