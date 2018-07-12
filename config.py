class Config(object):
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.embedding_size = 300
        self.hidden_size = 200
        self.filters = [3, 4, 5]
        self.num_filters = 256
        self.num_classes = 10
        self.max_length = max_length
        self.num_epochs = 20
        self.lr = 0.001
        self.dropout = 0.5
        self.attention = True