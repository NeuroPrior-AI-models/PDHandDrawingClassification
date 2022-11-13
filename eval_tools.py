import torch
import matplotlib.pyplot as plt


def zero_one_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)  # pred is a list of indices
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def plot_curves(history):
    val_loss = []
    train_loss = []
    train_acc = []
    val_acc = []
    time = list(range(len(history)))
    for h in history:
        train_loss.append(h['train_loss'])
        train_acc.append(h['train_acc'])
        # val_loss.append(h['val_loss'])
        # val_acc.append(h['val_acc'])

    # Plot loss vs epoch curve:
    # plt.plot(time, val_loss, c='red', label='val_loss', marker='x')
    plt.plot(time, train_loss, c='blue', label='train_loss', marker='x')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # Plot accuracy vs epoch curve:
    # plt.plot(time, val_acc, c='red', label='val_acc', marker='x')
    plt.plot(time, train_acc, c='blue', label='train_acc', marker='x')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
