from text_gnn.model import TextBiLSTM
from text_gnn.dataset import MyDataset
from text_gnn.config import TEMP_PATH, RECORD_PATH

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_eval(data_loader, cate):
    acc, loss_sum = 0, 0.0
    if cate == 'train':
        model.train()
    else:
        model.eval()
    for i, (x, target) in enumerate(data_loader):
        x, target = x.cuda(), target.cuda()
        y = model(x)
        if cate == 'train':
            optimizer.zero_grad()
            loss = loss_func(y, target)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_func(y, target)
        acc += sum(y.max(dim=1)[1].eq(target)).cpu().numpy().tolist()
        loss_sum += loss.cpu().detach().numpy().tolist()
    acc = acc * 100 / len(data_loader.dataset)
    loss_sum = loss_sum / len(data_loader)
    return acc, loss_sum


if __name__ == '__main__':
    num_words = 42725
    num_classes = 20
    embedding_dim = 300
    hidden_size = 150
    word2vec = np.load(TEMP_PATH + '/word2vector.npy')
    dropout = 0.2

    start = 54
    padding_len = 320
    batch_size = 512
    lr = 0.0001

    print("init & load...")
    train_data = DataLoader(MyDataset('train', padding_len), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(MyDataset('test', padding_len), batch_size=batch_size)
    model = TextBiLSTM(num_words, num_classes, embedding_dim, hidden_size, word2vec, dropout)
    if start != 0:
        model.load_state_dict(torch.load(RECORD_PATH + '/model.{}.pth'.format(start)))
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("start...")
    model = model.cuda()
    for epoch in range(start + 1, 1000):
        t1 = time.time()
        train_acc, train_loss = train_eval(train_data, 'train')
        test_acc, test_loss = train_eval(test_data, 'test')
        cost = time.time() - t1
        print("epoch=%s, cost=%.2f, train:[loss=%.4f, acc=%.2f%%], test:[loss=%.4f, acc=%.2f%%]"
              % (epoch, cost, train_loss, train_acc, test_loss, test_acc))
        torch.save(model.state_dict(), RECORD_PATH + '/model.{}.pth'.format(epoch))
