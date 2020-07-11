import os
import torch


# home_dir = "gdrive/My Drive/propaganda_detection"
home_dir = "./"
data_dir = os.path.join(home_dir, "datasets")
model_dir = os.path.join(home_dir, "model_dir")

# distinct_techniques = list(set([y for x in techniques for y in x])) # idx to tag
distinct_techniques = [
 'Flag-Waving',
 'Name_Calling,Labeling',
 'Causal_Oversimplification',
 'Loaded_Language',
 'Appeal_to_Authority',
 'Slogans',
 'Appeal_to_fear-prejudice',
 'Exaggeration,Minimisation',
 'Bandwagon,Reductio_ad_hitlerum',
 'Thought-terminating_Cliches',
 'Repetition',
 'Black-and-White_Fallacy',
 'Whataboutism,Straw_Men,Red_Herring',
 'Doubt'
]
distinct_techniques.insert(0, 'Non_propaganda')
tag2idx = {t: i for i, t in enumerate(distinct_techniques)}

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_ARTICLES = 20
LANGUAGE_MODEL = "Roberta"

if LANGUAGE_MODEL == "Roberta":
  from transformers import RobertaTokenizer, RobertaForSequenceClassification
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', lower_case=True)
  pretrained_model = RobertaForSequenceClassification.from_pretrained(
      "roberta-base",
      num_labels = len(distinct_techniques),
      output_attentions = False, 
      output_hidden_states = False,
  )

else:
  from transformers import BertTokenizer, BertForSequenceClassification
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', lower_case=True)
  pretrained_model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      num_labels = len(distinct_techniques),
      output_attentions = False, 
      output_hidden_states = False,
  )
