# Model

This repository contains the code to train models.

## Installation

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

## Database setup

The model training script expects ngram data to be stored in `docs.doc_ngram_0` (`docs` schema, `doc_ngram_0` table) within a postgres database. For the script to connect to this database, add a .env file in the `/src` directory that sets the `PG_URI` variable the [connection string](https://stackoverflow.com/questions/3582552/what-is-the-format-for-the-postgresql-connection-string-url) for the database.

An example .env file could be:

```sh
PG_URI=postgres://<username>:<password>@localhost/<database-name>
```

Where `<username>`, `<password>`, and `<database-name>` are replaced with the values for connecting to the database on your server.

## Helpful Notes for Later
- `m.wv.most_similar(negative=['man'], positive=['dad', 'woman'])`
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#visualising-word-embeddings