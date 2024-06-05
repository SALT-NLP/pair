import json, os, re
import pandas as pd
import numpy as np
import datasets
from datasets.utils import disable_progress_bar
from datasets.utils.logging import set_verbosity
from glob import glob
from collections import defaultdict
from src.utils import load_wiki_balance
from ast import literal_eval

def print_table_1_left():
    fit_unique_topics = set([x.split('_')[0] for x in fit_qrels.keys()])
    unique_topics = set([x.split('_')[0] for x in qrels.keys()])
    
    fit_categories = categories_csv[[str(_id) in fit_unique_topics for _id in categories_csv['id'].values]]
    fit_domains = set()
    for cats in fit_categories.categories:
        fit_domains.update(set(cats))
        
    categories = categories_csv[[str(_id) in unique_topics for _id in categories_csv['id'].values]]
    domains = set()
    for cats in categories.categories:
        domains.update(set(cats))
    
    print(f"""TABLE 1 (Left)
\t\tSynthetic\tNatural
Domains\t\t{len(fit_domains)}\t\t{len(domains)}
Topics\t\t{len(fit_unique_topics)}\t\t{len(unique_topics)}
Queries\t\t{len(fit_qrels)}\t\t{len(qrels)}
Documents\t{len(fit_corpus)}\t\t{len(corpus)}
Google Search\t✗\t\t✓
Gold Labels\t✓\t\t✗
Applies: rND\t✓\t\t✗
Applies: rKL\t✓\t\t✗
Applies: DUO\t✓\t\t✓""")
    
def print_table_1_right():
    total_topics = 0

    abbreviations = {
        'Entertainment': 'Entertain.',
        'History': 'History',
        'Law and order': 'Law',
        'Media and culture': 'Culture',
        'Politics and economics': 'Politics',
        'Religion': 'Religion',
        'Sex, sexuality, and gender identity': 'Sexuality',
        'Sports': 'Sports',
        'Environment': 'Environ.',
        'Languages': 'Languages',
        'People': 'People',
        'Philosophy': 'Philosophy',
        'Psychiatry': 'Psychiatry',
        'Science, biology, and health': 'Science',
        'Technology': 'Technology'
    }

    print("TABLE 1 (Right)")
    for cat in sorted(cats):

        quality_fn = "hit/quality_audit_synthetic/" + "_".join(cat.replace(', ', " ").split(" ")) + ".csv"
        if os.path.exists(quality_fn):
            df = pd.read_csv(quality_fn)

            results = df.mean()

            consider = categories_csv[[cat in c for c in categories_csv['categories'].values]][['id', 'key', 'categories']].copy()
            consider_queries = [ q for q in list(fit_qrels.keys()) for _id in consider['id'].values if q.split('_')[0]==str(_id)]
            consider_docs = [ c for c in list(fit_corpus.keys()) for _id in consider['id'].values if c.split('_')[0]==str(_id)]


            num_topics = str(len(consider))
            num_queries = str(len(consider_queries))
            num_docs = str(len(consider_docs))
            total_topics += int(num_topics)
            print(' & '.join([abbreviations[cat], f"{int(num_topics):,}", 
                              f"{int(num_queries):,}", f"{int(num_docs):,}"]) + ' && ' + ' & '.join([
                              f"{results['Relevance_query']:.1f}",
                              f"{results['Subjectiveness_query']:.1f}",
                              f"{results['Faithfulness_doc']:.1f}",
                              f"{results['Coherence_doc']:.1f}",
                              f"{results['Relevance_doc']:.1f}",
                              f"{results['Fluency_doc']:.1f}"
                             ]) + "\\\\")

    for cat in sorted(cats):

        quality_fn = "hit/quality_audit_synthetic/" + "_".join(cat.replace(', ', " ").split(" ")) + ".csv"
        if not os.path.exists(quality_fn):

            consider = categories_csv[[cat in c for c in categories_csv['categories'].values]][['id', 'key', 'categories']].copy()
            consider_queries = [ q for q in list(fit_qrels.keys()) for _id in consider['id'].values if q.split('_')[0]==str(_id)]
            consider_docs = [ c for c in list(fit_corpus.keys()) for _id in consider['id'].values if c.split('_')[0]==str(_id)]

            num_topics = str(len(consider))
            num_queries = str(len(consider_queries))
            num_docs = str(len(consider_docs))
            total_topics += int(num_topics)
            print(' & '.join([abbreviations[cat], f"{int(num_topics):,}", 
                              f"{int(num_queries):,}", f"{int(num_docs):,}"]) + ' && ' + ' & '.join(['-']*6) + "\\\\")

set_verbosity(50)
disable_progress_bar()
    
categories_csv = pd.read_csv("categories.csv")
categories_csv["categories"] = [literal_eval(x) for x in categories_csv["categories"].values]
cats = set()
for c in categories_csv["categories"].values:
    cats.update(set(c))
corpus, queries, qrels = load_wiki_balance(subset='natural')
fit_corpus, fit_queries, fit_qrels = load_wiki_balance(subset='synthetic')

print("Generating Tables...")
print()
print_table_1_left()
print()
print_table_1_right()