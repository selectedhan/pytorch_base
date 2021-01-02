import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import preprocessing

def train(config) :
    # 변수 사용은 config.lr
    trainset = preprocessing.MyDataset(x_data, y_data, transform=preprocessing.MyTransform_numpy)
    # testset = preprocessing

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_worker=1)
    # testloader = DataLoader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.model().to(device)
    criterion = nn.CrossEntrypyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), le=config.lr)

    print('START TRAINING')
    for epoch in range(config.epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % config.log_step == 0:
                if config.save_model_in_epoch:
                    torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, i + 1, len(train_loader), loss.item()))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, avg_epoch_loss))
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argumnet('--model_path', type=str, default='', choices=[])
    parser.add_argumnet('--model_saved', type=str, default='')

    parser.add_argument('--batch_size', type=int, default=)
    parser.add_argument('--epoch', type=int, default=)
    parse.add_argument('--lr', type=int, default=)
    parser.add_argument('--s', '--save_model_in_epoch', action='store_true')
    config = parser.parse_args()
    print(config)

    train(config)