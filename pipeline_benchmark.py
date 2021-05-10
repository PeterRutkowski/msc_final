import numpy as np
import torch
import torch.optim as optim

x_train = np.load('data/x_train_none_none.npz', allow_pickle=True)['data']
y_train = np.load('data/y_train.npz', allow_pickle=True)['data']

N, D_in, D_out = x_train.shape[0], x_train.shape[1], 10
learning_rate = 0.001

x_train = torch.Tensor(x_train)
y_train = np.squeeze(torch.LongTensor(y_train))

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 1000),
    torch.nn.Sigmoid(),
    torch.nn.Linear(1000, D_out),
)

loss_fn = torch.nn.NLLLoss()
losses = []
batches = 256
n_batches = int(N / batches)
epochs = 50

train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batches,
                                           shuffle=True, num_workers=1)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(epochs):
    print(epoch)
    running_loss = 0.0
    for i, datap in enumerate(train_loader, 0):
        inputs, labels = datap
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % n_batches == n_batches - 1:
            running_loss = 0.0
    scheduler.step()

torch.save(model, 'data/nn_benchmark.pt')
