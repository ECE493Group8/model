# Database Notes

On the ISAIC vm, there is postgres DB with the first 10 million rows of the first chunk of ngrams.

DB Name: `malamud`
Table: `docs.doc_ngrams_0`

## Creating DB from file

Command to load the first ~10mil lines from the ngram dump into the `malamud` db:

```
sed -n '44,100000000p;100000000q' doc_ngrams_0.sql | psql malamud
```