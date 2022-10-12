import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')


def output_filepath_exists():
    return os.path.exists('outputs/')


def save_model(epochs, model, optimiser, criterion):
    """
    Save model to disk
    :param epochs: number of epochs
    :param model: our CNN
    :param optimiser: the chosen optimiser
    :param criterion: the chosen loss function
    :return: None
    """
    print('Model name: ', model.name)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss': criterion,
    }, f'outputs/{model.name}_model_{epochs}_epochs.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name, epochs):
    """
    Saves plots describing accuracy and loss over the epochs
    :param train_acc: training accuracy
    :param valid_acc: validation accuracy
    :param train_loss: training loss
    :param valid_loss: validation loss
    :param model_name: name of the model
    :param epochs: number of epochs
    :return:
    """

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'outputs/{model_name}_accuracy_{epochs}_epochs.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{model_name}_loss_{epochs}_epochs.png')


