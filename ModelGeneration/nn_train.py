
import torch
import torchvision
import torch.optim as optim
from torch import Tensor

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from Nets import SimpleNet



def train(inp_dim, net, X_train, y_train, num_epochs, eps, width, num_layers, optimizer_name, lr, batch_size, weight_decay, momentum=0, verbose=False):
    X_train = Tensor(X_train)
    y_train = Tensor(y_train)

    out_dim = 1
    if net is None:
        net = SimpleNet(inp_dim, out_dim, width, num_layers, dropout_p=0, activation="relu")
        
    train_dataloader = DataLoader(TensorDataset(X_train, y_train.unsqueeze(1)), batch_size=batch_size,
                              pin_memory=False, shuffle=True)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', threshold=1, threshold_mode="rel", patience=10, factor=0.5)

    criterion = torch.nn.MSELoss(reduction="mean")

    loss_res = []

    for num_epoch in tqdm(range(num_epochs), disable=(not verbose)):
        avg_loss = 0
        num_loss = 0
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32)
            y_train = y_train.type(torch.float32)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(X_train)
            loss = criterion(outputs, y_train)
            loss.backward(retain_graph=True)
            optimizer.step()
            avg_loss += loss
            num_loss += 1

        avg_loss = avg_loss/num_loss
        loss_res.append(avg_loss)
        if avg_loss < eps:
            break

        # scheduler.step(avg_loss)
        # print(scheduler.optimizer.param_groups[0]['lr'])
    print(loss_res)
    return net, loss_res