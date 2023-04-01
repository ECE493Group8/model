import polars as pl

csv_base_fn = 'docs.doc_ngram_0_X.csv'

num_csvs = 16

for i in range(1, num_csvs+1):
    csv_fn = csv_base_fn.replace('X', str(i))
    parq_fn = csv_fn.replace('.csv', '.parquet')
    print(f'Converting {csv_fn} to {parq_fn}')
    df = pl.read_csv(csv_fn)
    df.write_parquet(parq_fn)