from collections import Counter
import numpy as np
import random


class Vocab(object):
    def __init__(self, vocab_file, max_vocab_size):
        self._word2idx = dict()
        self.MARKS = ["PAD", "UNK"]
        self._size = 0
        for word in self.MARKS:
            idx = len(self._word2idx)
            self._word2idx[word] = idx
            self._size += 1

        assert vocab_file is not None
        # vocab file is consist of word + \t + freq for each line
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                token = line.split("\t")
                word = token[0]
                if word not in self._word2idx:
                    idx = len(self._word2idx)
                    self._word2idx[word] = idx
                    self._size += 1
                if self._size > max_vocab_size:
                    break

    @staticmethod
    def build_vocab_file(input_file, output_file):
        fr = open(input_file, 'r', encoding="utf-8")
        words = []
        for i, doc in enumerate(fr.readlines()):
            # split document into word level
            i += 1
            tokens = doc.split()
            if i % 1000 == 0:
                print(i)
            words.append(tokens)
            # words = np.concatenate([words, tokens], axis=0)
        fr.close()
        words = np.concatenate(words, axis=0)
        counter = Counter(words)
        words_freqs = counter.most_common()
        with open(output_file, "w", encoding="utf-8") as f:
            for word, freq in words_freqs:
                f.write(word + "\t" + str(freq) + "\n")

    def word2idx(self, word):
        # map word to corresponding idx
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx["UNK"]


def load_data(data_file, label_file, vocab: Vocab, max_length=None):
    data = open(data_file, "r", encoding="utf-8").readlines()
    labels = open(label_file, "r", encoding="utf-8").readlines()
    # tokenize document into word level
    tokenized_docs = list(
        map(lambda doc: doc.strip().replace("</s>", "").split(), data))

    # change string label into integer and make
    # range 0 ~ 9 from 1 ~ 10
    labels = list(map(lambda label: int(label.strip()) - 1, labels))
    # change token into idx and zero pad
    vectorized_docs, sequence_lengths, length = _vectorize(tokenized_docs,
                                                           vocab,
                                                           max_length)
    return vectorized_docs, sequence_lengths, length, labels


def _vectorize(docs, vocab, max_length):
    # change tokens into indices and zero pad
    vectorized_corpus = []
    sequence_lengths = []
    for doc in docs:
        indices = list(map(lambda word: vocab.word2idx(word), doc))
        if max_length:
            indices = indices[:max_length]
        vectorized_corpus.append(indices[:max_length])
        sequence_lengths.append(len(indices))
    if max_length is None:
        max_length = max(sequence_lengths)
    vectorized_corpus = _zero_pad(vectorized_corpus, vocab, max_length)
    return vectorized_corpus, sequence_lengths, max_length


def _zero_pad(docs, vocab, max_length):
    zero_padded = list(map(
        lambda doc: doc + [vocab.word2idx("PAD")] * (max_length - len(doc)),
        docs))
    return zero_padded


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx: min(length, start_idx + batch_size)]


if __name__ == "__main__":
    Vocab.build_vocab_file("data/train/reviews.txt", "data/vocab")
