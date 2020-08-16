import pandas as pd


class dataset:
    def __init__(self):
        self.data = pd.read_table('./dataset/train.txt', delimiter=' ',
                                  names=['sp', 'tp', 'pg', 'p'])
        # seq_len = sos + max_len + eos
        self.max_len = self.data.applymap(lambda x: len(x)).max().max()
        self.idx_list = list(range(self.data.count().sum()))

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        row = idx % self.data.count()[0]
        col = idx // self.data.count()[0]
        word = self.data.iloc[row, col]
        # sp = self.data.iloc[row, 0]
        # return word with tense, tense idx, simple present of word
        return word, col


def main():
    pass


if __name__ == "__main__":
    main()
