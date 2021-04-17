import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

def component_model(x_train, y_train, x_test, exp, i):
    N, D_in, H1, H2, D_out = x_train.shape[0], x_train.shape[1], 2000, 1500, 2
    learning_rate = 0.001

    x = torch.Tensor(x_train)
    y = np.squeeze(torch.LongTensor(y_train))

    model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.Sigmoid(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out),
    torch.nn.LogSoftmax(dim = 1),
    )

    loss_fn = torch.nn.NLLLoss()

    batchs = 256

    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchs,
                                            shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    batchesn = int(N/batchs)
    epochs = 50

    for epoch in range(epochs):
        running_loss = 0.0
        for i, datap in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = datap
            inputs = inputs.to('cpu', non_blocking=True)
            labels = labels.to('cpu', non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    scheduler.step()

    n_images = x_test.shape[0]
    x_test = torch.Tensor(x_test)

    model.eval()
    
    torch.save(model, 'experiments/BRP_NN_{}/model{}.pt'.format(exp, i))

    return np.exp(model(x_test).detach().numpy())[:, 1]

def BRP(exp):
    l = np.load('data/in10_split_converted.npz',
              allow_pickle=True)
    x_train, x_test = l['x_train'], l['x_test_none']

    l = np.load('experiments/{}.npz'.format(exp),
              allow_pickle=True)
    y_train = l['x_train']

    y_pred = []
    for i in tqdm(range(y_train.shape[1])):
        prediction = component_model(x_train, y_train[:, i], x_test, exp, i)
        y_pred.append(prediction)
        np.savez('experiments/BRP_NN_{}/rep'.format(exp), y_pred=np.asarray(y_pred).T)
        
BRP('comp120_pca_dbscan60')
