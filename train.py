# inspo: https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time

from tqdm.auto import tqdm
from model import CNNModel, ViTLightningModule
from datasets import train_loader, valid_loader
from utils import save_model, save_plots
from web_scrape import input_filepath_exists, generate_dataset


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('-r', '--redownload', type=bool, default=False,
                    help='run the image scraping script')
parser.add_argument('-m', '--model', type=str, default='CNN',
                    help='choose which model to run from the following: {CNN, ViT}')
args = vars(parser.parse_args())

lr = 1e-3
epochs = args['epochs']
redownload = args['redownload']
model_flag = args['model']

if redownload and input_filepath_exists():
    print('Webscraping images')
    generate_dataset()


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

if model_flag == 'CNN':
    model = CNNModel().to(device)
elif model_flag == 'ViT':
    model = ViTLightningModule().to(device)
else:
    model = CNNModel().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params} total parameters")

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params} training parameters")

optimiser = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, optimiser, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimiser.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(test_loader.dataset))
    return epoch_loss, epoch_acc


train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# for epoch in range(epochs):
#     print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#     train_epoch_loss, train_epoch_acc = train(model, train_loader, optimiser, criterion)
#     valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)
#     train_loss.append(train_epoch_loss)
#     valid_loss.append(valid_epoch_loss)
#     train_acc.append(train_epoch_acc)
#     valid_acc.append(valid_epoch_acc)
#     print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#     print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#     print('-' * 50)
#     time.sleep(5)
#
# save_model(epochs, model, optimiser, criterion)
# save_plots(train_acc, valid_acc, train_loss, valid_loss, model.name, epochs)

