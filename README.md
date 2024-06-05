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
        <a href="#validations-and-additional-experiments">Validations and Additional Experiments</a>
    <p>
</h4>

<!-- > The development of PAIR is supported by: -->

<h3 align="center">
    <a href="https://nlp.stanford.edu/"><img style="float: left; padding: 2px 7px 2px 7px;" height="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Stanford_Cardinal_logo.svg/314px-Stanford_Cardinal_logo.svg.png" /></a>
    <a href="https://www.cc.gatech.edu/"><img style="float: middle; padding: 2px 7px 2px 7px;" height="100" src="https://upload.wikimedia.org/wikipedia/commons/8/84/Georgia_Tech_logo_2021_Cropped.png" /></a>
    <a href="https://ai.meta.com/research/"><img style="float: right; padding: 2px 7px 2px 7px;" height="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg/2560px-Meta_Platforms_Inc._logo_%28cropped%29.svg.png" /></a>
</h3>

This repository contains data and code for the paper **Measuring and Addressing Indexical Bias in Information Retrieval**. For more information, please reach out to the authors:

<table>
  <tr>
    <td align="center"><a href="https://calebziems.com/"><img src="https://calebziems.com/assets/img/caleb_sf.jpeg" width="100px;" alt=""/><br /><sub><b>Caleb Ziems</b></sub></a></td>
    <td align="center"><a href="https://williamheld.com/"><img src="https://avatars.githubusercontent.com/u/9847335?v=4" width="100px;" alt=""/><br /><sub><b>William Held</b></sub></a></td>
    <td align="center"><a href="https://janedwivedi.github.io/"><img src="https://conference2023.mlinpl.org/images/optimized/speakers-2023-600x600/JaneDwivedi-Yu.webp" width="100px;" alt=""/><br /><sub><b>Jane Dwivedi-Yu</b></sub></a></td>
    <td align="center"><a href="https://cs.stanford.edu/~diyiy/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-1wX8XpndSuQulTQg8O8vh63T-9QD9jD-Lg&s" width="100px;" alt=""/><br /><sub><b>Diyi Yang</b></sub></a></td>
  </tr>
</table>

## *What is PAIR?*
:people_holding_hands: `PAIR` is designed to help you identify and mitigate indexical biases in your IR systems. :people_holding_hands: `PAIR` includes a set of evaluation metrics, data resources, and human subjects study interfaces that help you measure and experimentally understand the Search Engine Manipulation Effect.

## Setup

#### From Source
```bash
$ git clone https://github.com/SALT-NLP/pair.git
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

## Datasets
You can view the WikiBalance datasets on **[Hugging Face](https://huggingface.co/collections/SALT-NLP/pair-665f8ffd0b1c27cf149d3106)**.

| Dataset   | Huggingface Name | Gold Labels | Type | Topics  | Queries | Documents |
| -------- | -----| ---------| ------- | --------- | ----------- | ---------|
| WikiBalance Synthetic    | `SALT-NLP/wiki-balance-synthetic` | ❌ | ``test``|  1.4k   | 4k | 31.5k |
| WikiBalance Natural      | `SALT-NLP/wiki-balance-natural` | ✅ | ``test``|  288   | 452 | 4.6k |


## System Audits
You can replicate all system audits from Tables 4 and 5 in the paper by running the following script:

```bash
bash run_audit.sh
```

Only BM-25 and ColBERT require special setup to run. To set up ColBERT, follow the (BEIR demo instructions here)[https://github.com/beir-cellar/beir/tree/main/examples/retrieval/evaluation/late-interaction]. To run BM-25, use the following steps:

**On Mac**
1. Download `elasticsearch.zip` and unpack locally: [elastic.co/downloads/elasticsearch](https://www.elastic.co/downloads/elasticsearch)
2. Edit `config/elasticsearch.yml` to remove security features, setting `false` to `xpack.security.enabled`, `xpack.security.http.ssl.enabled`, `xpack.security.transport.ssl.enabled`
3. Move to the elasticsearch directory and run elasticsearch `bin/elasticsearch`
4. Run using `python -m src.modeling.run_bm25 --dataset "idea/wiki" --model "bm25"`
**On Linux**
Follow these instructions: [linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04](https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/)

## Validations and Additional Experiments
1. To print the summary tables from the paper, run `print_tables.py` from the main directory.
2. To replicate our metric validations in Table 2 (as well as Tables 6 and 7 in the Appendix), run `python -m src.experiments.metric_validation`
3. To replicate the SEME experiments, you can do the following:
   a. Re-run the experiments with your own participants using the HIT interface, `hit/seme/hit_pair_seme.html` OR
   b. Download the experimental data from (this Drive link)[https://drive.google.com/file/d/1TXKZueZFo_VbzMyui-V5YkQVvixysQuA/view?usp=drive_link] and place it in the `hit/seme` directory.
   c. Run `python -m src.experiments.seme_experiment.py`