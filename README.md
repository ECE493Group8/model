# Model

This repository contains the code to train models.

## Installation

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

# Preprocessing Data

The first step in training the model is to first preprocess the data. Follow the
instructions below to do so.

## Creating Phrases

We want our Word2Vec algorithm to train on data that treats common bigrams as a
single word. For example, to the Word2Vec model, the phrase "new york" is
represented as "new_york".

We have developed a pre-trained "phrase model" available in the `phraser/`
directory. It can be used as follows:

```python
>>> from gensim.models.phrases import FrozenPhrases
>>> phrases = FrozenPhrases.load("./phraser/frozen_phrases_merged.model")
>>> phrases[["new", "york"]]
['new_york']
```

This phrase model was trained on about 8.7 billion lines of the first ngram
slice of the Malamud Index.

The phrase model that comes with this repository is fairly comprehensive. If you
would like to develop your own phrase model, run the command:

```sh
$ python3 src/preprocess.py phrase \
  --n-processes <n-processes> \
  --skip-rows <skip-rows> \
  --rows-in-mem <rows-in-mem> \
  --save-path <save-path> \
  --data-path <data-path>
```

### Arguments

- `<n-processes>`: The number of processes used to load in data from memory and
  compute the phrases.
- `<skip-rows>`: The number of rows to skip before reading ngrams from the input
  file.
- `<n-rows>`: The number of rows to read in and develop phrases from in the
  input file.
- `<rows-in-mem>`: The number of rows each process keeps in memory to process at
  a time.
- `<save-path>`: The directory to save phrase models and logs to.
- `<data-path>`: The path to the Malamud Index ngrams (or keywords) file to
  process.

### Example

```sh
$ python3 src/preprocess.py phrase \
  --n-processes 16 \
  --skip-rows 45 \
  --n-rows 22336970000 \
  --rows-in-mem 10000000 \
  --save-path /storage8TB/phrase_model \
  --data-path /storage8TB/malamud/doc_ngrams/doc_ngrams_0.sql
```

## Creating Preprocessed Data Parquets

Once a phrase model is developed (or simply used from the `phraser/` directory),
the phraser model can be used to preprocess data from an input ngrams or
keywords file. In addition to combining common bigrams, the preprocessing will
also remove unnecessary punctuation and stopwords. The preprocessed data is then
stored in a parquet file which represents a dataframe with one column,
"keywords_lc" of lists of strings for each preprocessed item.

```sh
$ python3 src/preprocess.py parquet \
  --skip-rows <skip-rows> \
  --n-rows <n-rows> \
  --save-path <save-path> \
  --save-tag <save-tag> \
  --data-path <data-path>
```

> Note that by default, the phrase model in the `phraser/` directory is used. You
can change this by changing the path to the phrase model,
`DEFAULT_FROZEN_PHRASES_MODEL`, in `preprocessor.py`.

### Arguments

- `<skip-rows>`: The number of rows to skip before reading ngrams from the input
  file.
- `<n-rows>`: The number of rows to read in and develop phrases from in the
  input file.
- `<save-path>`: The directory to save phrase models and logs to.
- `<save-tag>`: The name of the file to save the parquet and log file.
- `<data-path>`: The path to the Malamud Index ngrams (or keywords) file to
  process.

### Example

```sh
$ python3 src/preprocess.py parquet \
  --skip-rows 45 \
  --n-rows 2400000000 \
  --save-path /storage8TB/keywords_parquets \
  --save-tag docs_keywords_0 \
  --data-path /storage8TB/malamud/doc_keywords/doc_keywords_0.sql
```

# Training

Once the data is preprocessed, one can train the Word2Vec model using the
following command:

```sh
$ python3 src/train.py \
  --directory <directory> \
  --workers <workers> \
  --epochs <epochs> \
  --vector_size <vector_size> \
  --parquet_path <parquet_path> \
  --column <column> \
  --build_vocab_progress_per <progress_per>
```

### Arguments

- `<directory>`: The directory to save the Word2Vec models and training logs.
- `<workers>`: The number of processes to use for training.
- `<epochs>`: The number of epochs to run during training.
- `<vector_size>`: The number of dimensions each word vector will have.
- `<parquet_path>`: The file pattern of the parquet files to train on.
- `<column>`: The column of the dataframe to train on.
- `<progress_per>`: Log to file after processing this number of sentences when
  building the vocabulary.

### Example

```sh
$ python3 src/train.py \
  --directory ./train_keywords_0_300vec \
  --workers 16 \
  --epochs 5 \
  --vector_size 300 \
  --parquet_path "/mnt/doc_keywords_parquets/*.parquet" \
  --column keywords_lc \
  --build_vocab_progress_per 10000000
```

# Testing

We use Python's `unittest` testing library and the code coverage library
`coverage` to ensure 100% line coverage.

Run the tests using the command:

```sh
$ python3 -m coverage run -m unittest discover
```

Then, verify the code coverage with the command:

```sh
$ python3 -m coverage report
```

It should show that all tested files have at least 100% line coverage. Our
testing document outlines other testing-related guarantees, such as equivalence
class testing, of our tests in this repository.
