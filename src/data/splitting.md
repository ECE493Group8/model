To split csv:

`cat docs.doc_ngrams_0.csv | parallel --header : --pipe -N40874 'cat >docs.doc_ngram_0_{#}.csv'`

Where the 40874 is ceiling((num lines in file)/(desired num of chunks))

Had to install parallel with `sudo apt install parallel`