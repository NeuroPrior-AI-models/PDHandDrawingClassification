import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from dataset import PDHandDrawingDataset
from transformations import train_transform, val_transform, test_transform
from ResNeXt32 import PDDrawingPretrainedResnext32
from devices import get_default_device, to_device, DeviceDataLoader
from debug_tools import try_batch
from train import evaluate, fit_one_cycle
from eval_tools import plot_curves

DEBUG = 0


if __name__ == '__main__':
    # Load datasets:
    dataset = ImageFolder("C:/Users/PaulS/Desktop/PDHandDrawing/spiral_restructured")
    print(f'Dataset has {len(dataset)} examples and {len(dataset.classes)} classes.')

    # Split dataset into training and test sets:
    test_size = int(len(dataset) * 0.3)
    train_size = len(dataset) - test_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    print(f'Training size: {len(train_ds)}, Test size: {len(test_ds)}.')

    if DEBUG:   # examine an image
        img, label = train_ds[5]
        print(dataset.classes[label])
        plt.imshow(img)
        plt.show()
        print(type(img))

    # Create dataset classes and specify transformations:
    train_dataset = PDHandDrawingDataset(train_ds, train_transform)
    test_dataset = PDHandDrawingDataset(test_ds, test_transform)

    # Create DataLoader classes:
    batch_size = 32
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size * 2, num_workers=2, pin_memory=True)

    # Create model:
    resnext32_pretrained = PDDrawingPretrainedResnext32()

    # # Move dataloader and model to GPU:
    # device = get_default_device()
    # print(f'Using device {device}')
    #
    # train_dl = DeviceDataLoader(train_dl, device)
    # test_dl = DeviceDataLoader(test_dl, device)
    # to_device(resnext32_pretrained, device)

    if DEBUG:   # feed forward a batch to check output
        try_batch(resnext32_pretrained, train_dl)

    if DEBUG:   # check loss and accuracy prior to any training
        print(evaluate(resnext32_pretrained, train_dl))

    # Train model:
    history = fit_one_cycle(
        epochs=10,
        max_lr=0.001,
        model=resnext32_pretrained,
        train_loader=train_dl,
        opt_func=torch.optim.Adam
    )

    # Print model accuracy:
    print(f'Test accuracy: {evaluate(resnext32_pretrained, test_dl)}.')

    # Plot accuracy and loss curves during training:
    print(f'Metrics that could be plotted: {history[0].keys()}.')

    plot_curves(history)

    # Save model:
    torch.save(resnext32_pretrained.state_dict(), "models/resnext32_pretrained.pth")

