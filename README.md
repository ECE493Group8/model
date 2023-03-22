# Model

This repository contains the code to train models.

# Installation

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

# Helpful Notes for Later
- `m.wv.most_similar(negative=['man'], positive=['dad', 'woman'])`
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#visualising-word-embeddings

# TODO before training

- Change hyperparameters

# Database

Command for postgres executable:
```sh
export PATH=${PATH}:/usr/lib/postgresql/14/bin
```

## Installing `psycopg2`
- If you run into an error about libpq:
  https://stackoverflow.com/questions/58961043/how-to-install-libpq-fe-h


Statistics of 10M subset:
- number of ngrams with just one: 20725634
- number of ngrams: 99999955

A lot of this information can be found here:
https://ia802307.us.archive.org/18/items/GeneralIndex/data/README.txt

## Keywords

File: `doc_keywords_0.sql`
- `dkey`: The document key (hash of the document)
- `keywords`: The keywords
- `keywords_lc`: The keywords in lowercase (1 to 5grams)
- `keywords_tokens`: The number of keywords
- `keyword_score`: YAKE score of how meaningful the word is in the document, the
  the value, the more meaningful
- `doc_count`: Always 1 (for analytical purposes)
- `insert_date`: Date the record was inserted, initial load has NULL insert date

There seem to be a list of words in alphabetical [order for each paper](#from-keywords)


```sh
PG_URI=postgres://<username>:<password>@localhost/<database-name>
```

## N Grams

File: `doc_ngrams_0`
- `dkey`: The document key (hash of the document)
- `ngram`: The ngram
- `ngram_lc`: The ngram in lowercase
- `ngram_tokens`: The number of words in the ngram
- `ngram_count`: 
- `term_freq`: Number of occurences of the ngram in the document
- `doc_count`: Always 1 (for analytical purposes)
- `insert_date`: Date the record was inserted, initial load has NULL insert date

Is there punctuation?

Do we need ngrams with only one word?
