dev_folder = "datasets/dev-articles" # check that the path to the datasets folder is correct, if not adjust these variables accordingly 
propaganda_techniques_file = "scorers/propaganda-techniques-names-semeval2020task11.txt" # propaganda_techniques_file is in the tools.tgz file (download it from the team page)
task_SI_output_file = "baseline-output-SI.txt"

#
# Baseline task SI: create random spans for the development set
#
#
#

import glob
import os.path
import random

random.seed(10) # to make runs deterministic

# loading articles' content from *.txt files in the dev folder
file_list = glob.glob(os.path.join(dev_folder, "*.txt"))
articles_content, articles_id = ([], [])
for filename in file_list:
    with open(filename, "r", encoding="utf-8") as f:
        articles_content.append(f.read())
        articles_id.append(os.path.basename(filename).split(".")[0][7:])

with open(propaganda_techniques_file, "r") as f:
    propaganda_techniques_names = [ line.rstrip() for line in f.readlines() ]

with open(task_SI_output_file, "w") as fout:
    for article_content, article_id in zip(articles_content, articles_id):
        start_fragment, end_fragment, article_length = (0, 0, len(article_content))
        current_article_annotations = []
        while end_fragment < article_length:
            if end_fragment > 0:
                # check that there is no other annotation for the same anrticle and technique that overlaps
                intersection_length = 0
                if len(current_article_annotations) > 0:
                    span_annotation = set(range(start_fragment, end_fragment))
                    intersection_length = sum( [ len(span_annotation.intersection(previous_fragment))
                             for previous_fragment in current_article_annotations ])
                if len(current_article_annotations) == 0 or intersection_length > 0:
                    fout.write("%s\t%s\t%s\n" % (article_id, start_fragment, end_fragment))
                    current_article_annotations.append(set(range(start_fragment, end_fragment)))
            start_fragment += random.randint(0, max(1, article_length-start_fragment))
            end_fragment = min(start_fragment + random.randint(1,25), article_length)
        print("article %s: added %d fragments" % (article_id, len(current_article_annotations)))    

print("Predictions written to file " + task_SI_output_file)
