import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import os
import json
from tqdm import tqdm

from models.two_person_model import resnet, vgg
from two_person_data import PISCDataset


def get_splitList():
    with open('/PATH/PISC/relationship_split/relation_trainidx.json') as j:
        train_list = json.load(j)

    with open('/PATH/PISC/relationship_split/relation_validx.json') as j:
        val_list = json.load(j)

    with open('/PATH/PISC/relationship_split/relation_testidx.json') as j:
        test_list = json.load(j)

    test_list += val_list

    return train_list, test_list


if __name__ == '__main__':
    num_epochs = 30
    save_epoch = 10

    cudnn.enabled = True
    cudnn.benchmark = True

    # Import Data List
    train_list, test_list = get_splitList()

    train_datasets = PISCDataset('/PATH/PISC/image/data/preprocessed/train/', train_list)
    test_datasets = PISCDataset('/PATH/PISC/image/data/preprocessed/test/', test_list)

    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=False)

    ############# You can change the model. ###########
    model = resnet()

    # Using Multiple GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU activate')
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for i, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x1, x2)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch + 1, num_epochs,
                                                                             i + 1, len(train_loader),
                                                                             running_loss / 100))
                running_loss = 0.0

        if epoch % save_epoch == save_epoch - 1:
            save_file = os.path.join('./checkpoint/', str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_file)

    # Evaluation
    for j in range(int(num_epochs / save_epoch)):
        num_model = save_epoch * (j + 1)

        ########### Modify the checkpoint directory path ###########
        save_file = os.path.join('./checkpoint/', str(num_model) + '.pth')

        ############# You can change the model. ###########
        model = resnet()
        model = model.to(device)

        loaded_params = torch.load(save_file)
        new_params = model.state_dict().copy()
        for i in loaded_params:
            i_parts = i.split('.')
            if i_parts[0] == 'module':
                new_params['.'.join(i_parts[1:])] = loaded_params[i]
            else:
                new_params[i] = loaded_params[i]
        model.load_state_dict(new_params)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.eval()

        correct = 0.0
        with torch.no_grad():
            for k, (x1, x2, y) in enumerate(train_loader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                outputs = model(x1, x2)

                _, predicted = torch.max(outputs, 1)
                correct += torch.sum(predicted == y.data)

        epoch_acc = 100 * correct / len(train_loader.dataset)
        print('Train Accuracy({:d}) {:.2f}%'.format(num_model, epoch_acc))

        correct = 0.0
        with torch.no_grad():
            for k, (x1, x2, y) in enumerate(test_loader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                outputs = model(x1, x2)

                _, predicted = torch.max(outputs, 1)
                correct += torch.sum(predicted == y.data)

        epoch_acc = 100 * correct / len(test_loader.dataset)
        print('Test Accuracy({:d}) {:.2f}%'.format(num_model, epoch_acc))
