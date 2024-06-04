# PAIR: Perspective-Aligned Information Retrieval

<p align="center">
    <a href="http://creativecommons.org/licenses/by-sa/4.0/">
        <img alt="License" src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="#">
        <img alt="Open Source" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://openreview.net/pdf?id=6buHk8B0ww">Paper</a> |
        <a href="#setup">Setup</a> |
        <a href="#quick-example">Quick Example</a> |        
        <a href="#datasets">Datasets</a> |
        <a href="#system-audits">System Audits</a> |
        <a href="#behavioral-experiments">Behavioral Experiments</a>
    <p>
</h4>

<!-- > The development of PAIR is supported by: -->

<h3 align="center">
    <a href="https://nlp.stanford.edu/"><img style="float: left; padding: 2px 7px 2px 7px;" width="220" height="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Stanford_Cardinal_logo.svg/314px-Stanford_Cardinal_logo.svg.png" /></a>
    <a href="https://www.cc.gatech.edu/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="250" height="90" src="https://upload.wikimedia.org/wikipedia/commons/8/84/Georgia_Tech_logo_2021_Cropped.png" /></a>
    <a href="https://ai.meta.com/research/"><img style="float: right; padding: 2px 7px 2px 7px;" width="320" height="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg/2560px-Meta_Platforms_Inc._logo_%28cropped%29.svg.png" /></a>
</h3>

This repository contains data and code for the paper **Measuring and Addressing Indexical Bias in Information Retrieval** by [Caleb Ziems](https://calebziems.com/), [William Held](https://williamheld.com/), [Jane Dwivedi-Yu](https://janedwivedi.github.io/), [Diyi Yang](https://cs.stanford.edu/~diyiy/)

## *What is PAIR?*
:people_holding_hands: `PAIR` is designed to help you identify and mitigate indexical biases in your IR systems. :people_holding_hands: `PAIR` includes a set of evaluation metrics, data resources, human subjects study interfaces, and 

## Setup

#### From Source
```bash
$ git clone https://github.com/cjziems/pair.git
$ cd pair
$ conda create -n pair python=3.9.16
$ conda activate pair
$ pip install -r requirements.txt
```

## Quick Example

You can run this example in the `Demo.ipynb` jupyter notebook. 

```python
from src.metrics.duo import Duo, get_relevant_corpus, get_relevant_corpus_retrieved, get_relevant_ranking
from src.utils import load_wiki_balance
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SentenceBERT
from beir.retrieval.evaluation import EvaluateRetrieval

# ----- RETRIEVAL -----
## load the WikiBias_Natural retrieval corpus
corpus, queries, qrels = load_wiki_balance(subset='natural')
## load an IR model from BEIR
retriever = EvaluateRetrieval(DRES(SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16))
## retrieve documents
retrieved = retriever.retrieve(corpus, queries)

from src.metrics.duo import Duo, get_relevant_corpus, get_relevant_corpus_retrieved, get_relevant_ranking
# ----- INDEXICAL BIAS EVALUATION -----
## initialize the metric 
d = Duo(embedding_model="sentence-t5-xl", step_size=1, random_state=7)

## load the synthetic corpus for fitting the Duo metric
fit_corpus, fit_queries, fit_qrels = load_wiki_balance(subset='synthetic')

## evaluate on the first query
query_idx = list(retrieved.keys())[0]

## embed documents to polarization scores
d.embed(transform_docs=get_relevant_corpus_retrieved(corpus, retrieved, query_idx, qrels), 
        fit_docs=get_relevant_corpus(fit_corpus, query_idx, fit_qrels),
       )

# compute DUO score
duo_score = d.Duo(ranking=get_relevant_ranking(retrieved, query_idx, qrels))
print(duo_score)
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://calebziems.com/"><img src="https://calebziems.com/assets/img/caleb_sf.jpeg" width="100px;" alt=""/><br /><sub><b>Caleb Ziems</b></sub></a></td>
    <td align="center"><a href="https://williamheld.com/"><img src="https://avatars.githubusercontent.com/u/9847335?v=4" width="100px;" alt=""/><br /><sub><b>William Held</b></sub></a></td>
    <td align="center"><a href="https://janedwivedi.github.io/"><img src="https://conference2023.mlinpl.org/images/optimized/speakers-2023-600x600/JaneDwivedi-Yu.webp" width="100px;" alt=""/><br /><sub><b>Jane Dwivedi-Yu</b></sub></a></td>
    <td align="center"><a href="https://cs.stanford.edu/~diyiy/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-1wX8XpndSuQulTQg8O8vh63T-9QD9jD-Lg&s" width="100px;" alt=""/><br /><sub><b>Diyi Yang</b></sub></a></td>
  </tr>
</table>