train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "datasets/train-task2-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
task_TC_output_file = "baseline-output-TC.txt"

#
# Baseline for Task TC
#
# Our baseline uses a logistic regression classifier on one feature only: the length of the sentence.
#
# Requirements: sklearn, numpy
#


from sklearn.linear_model import LogisticRegression
import glob
import os.path
import numpy as np
import sys
import codecs

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    """
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    """
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), begin of a fragment, 
    end of a fragment. 
    """
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels


def compute_features(articles, span_starts, span_ends):
    # only one feature, the length of the span
    return np.array([ int(sp_ends)-int(sp_starts) for sp_starts, sp_ends in zip(span_starts, span_ends) ]).reshape(-1, 1)


### MAIN ###

# loading articles' content from *.txt files in the train folder
articles = read_articles_from_file_list(train_folder)

# loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder 
ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))

# compute one feature for each fragment, i.e. the length of the fragment, and train the model
train = compute_features(articles, ref_span_starts, ref_span_ends)
model = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs")
model.fit(train, train_gold_labels)

# reading data from the development set
dev_articles = read_articles_from_file_list(dev_folder)
dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)

# computing the predictions on the development set
dev = compute_features(dev_articles, dev_span_starts, dev_span_ends)
predictions = model.predict(dev)

# writing predictions to file
with open(task_TC_output_file, "w") as fout:
    for article_id, prediction, span_start, span_end in zip(dev_article_ids, predictions, dev_span_starts, dev_span_ends):
        fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
print("Predictions written to file " + task_TC_output_file)
