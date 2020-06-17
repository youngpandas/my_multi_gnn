from torch import nn
import numpy as np
import torch

from text_lstm.config import TEMP_PATH


class TextBiLSTM(nn.Module):

    def __init__(self, num_words,
                 num_classes,
                 embedding_dim=300,
                 hidden_size=150,
                 word2vec=None,
                 dropout=0.2):
        super(TextBiLSTM, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, embedding_dim, num_words)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), False, num_words)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.fc(x.sum(dim=1))
        return x


if __name__ == '__main__':
    num_words = 42725
    num_classes = 20
    embedding_dim = 300
    hidden_size = 150
    word2vec = np.load(TEMP_PATH + '/word2vector.npy')

    model = TextBiLSTM(num_words, num_classes, embedding_dim, hidden_size, word2vec)
    x = torch.randint(0, num_classes, (64, 300))
    y = model(x)
    print(y.size())
