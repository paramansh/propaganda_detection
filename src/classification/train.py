import time
import classification.utils as utils
import classification.config as config
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

device = config.device

def train(model, train_dataloader, eval_dataloader, epochs=5):
  optimizer = AdamW(model.parameters(),lr = 3e-5,eps = 1e-8) # ler = 5e-5
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)
  loss_values = []

  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
      if step % 100 == 0 and not step == 0:
        elapsed = utils.format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
      b_input_ids = batch[0].to(device)
      b_labels = batch[1].to(device)
      b_input_mask = batch[2].to(device)
      b_lengths = batch[3].to(device)
      model.zero_grad()        
      outputs = model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask,
                    # lengths=b_lengths,
                      labels=b_labels)
      
      # The call to `model` always returns a tuple, so we need to pull the 
      # loss value out of the tuple.
      loss = outputs[0]

      total_loss += loss.item()

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      optimizer.step()

      scheduler.step() # TODO

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(utils.format_time(time.time() - t0)))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_labels, b_input_mask, b_lengths = batch
      with torch.no_grad():        
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
                        # lengths=b_lengths)
      
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      tmp_eval_accuracy = utils.flat_accuracy(logits, label_ids)
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(utils.format_time(time.time() - t0)))
  print("")
  print("Training complete!")
