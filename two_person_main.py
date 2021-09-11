import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn

import os
from tqdm import tqdm

from models.two_person_model import resnet
from two_person_data import PISCDataset


if __name__ == '__main__':
    num_epochs = 30
    save_epoch = 10

    cudnn.enabled = True
    cudnn.benchmark = True

    datasets = PISCDataset('./crop')

    train_size = int(len(datasets) * 0.8)
    test_size = len(datasets) - train_size
    train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

    # Train, Test 클래스 별로 데이터 갯수 카운팅 / Train data label 리스트로 저장
    train_cnt = [0] * 6
    test_cnt = [0] * 6
    labels = []
    for i, datas in enumerate(train_dataset):
        x1, x2, y = datas
        train_cnt[y] += 1
        labels.append(y)
    else:
        print(f'Train Data : {sum(train_cnt)} : {train_cnt}')

    for i, datas in enumerate(test_dataset):
        x1, x2, y = datas
        test_cnt[y] += 1
    else:
        print(f'Test Data : {sum(test_cnt)}/* : {test_cnt}')

    num_samples = sum(train_cnt)

    class_weights = [num_samples / train_cnt[i] for i in range(len(train_cnt))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    sam_cnt = [0] * 6
    for i, datas in enumerate(train_loader):
        x1, x2, labels = datas
        for label in labels:
            sam_cnt[label] += 1
    else:
        print(f'Sampling Train Data : {sum(sam_cnt)} : {sam_cnt}')
        
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
        for i, datas in enumerate(train_loader):
            x1, x2, y = datas
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x1, x2)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()
            
        if epoch % save_epoch == save_epoch - 1:
            save_file = os.path.join('./checkpoint/', str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_file)

    ''' Validation '''
    save_file = os.path.join('./checkpoint/', str(num_epochs) + '.pth')
    model = resnet()

    load_params = torch.load(save_file)
    new_params = model.state_dict().copy()
    for j in load_params:
        jj = j.split('.')
        if jj[0] == 'module':
            new_params['.'.join(jj[1:])] = load_params[j]
        else:
            new_params[j] = load_params[j]
    model.load_state_dict(new_params)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

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
    print('Train Accuracy : {:.2f}%'.format(epoch_acc))

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
    print('Test Accuracy : {:.2f}%'.format(epoch_acc))
