from gensim.models.word2vec import LineSentence


class SmallDataset:
    """
    A generator to load the dataset into memory a part at a time.
    """
    def __init__(self, path: str):
        pass

    def __iter__(self):
        pass


if __name__ == "__main__":
    ls = LineSentence("./test_dataset.txt")
    for line in ls:
        print(line)
