import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_tools import zero_one_accuracy


class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.cross_entropy(out, targets)
        return loss

    # evaluate training accuracy for a batch
    def train_acc_step(self, batch):
        img, targets = batch
        out = self(img)
        acc = zero_one_accuracy(out, targets)
        return acc.detach()

    # evaluate training accuracy
    def train_acc_epoch_end(self, outputs):
        epoch_acc = torch.stack(outputs).mean()
        return epoch_acc.item()

    # evaluate accuracy and loss for a batch
    def eval_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.cross_entropy(out, targets)
        acc = zero_one_accuracy(out, targets)
        return {'acc': acc.detach(), 'loss': loss.detach()}

    # evaluate accuracy and loss
    def eval_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    # print result end epoch
    def epoch_end(self, epoch, result):
        # print(
        #     "Epoch [{}] : train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        #         epoch,
        #         result["train_loss"],
        #         result["train_acc"],
        #         result["val_loss"],
        #         result["val_acc"]
        #     )
        # )
        print(
            "Epoch [{}] : train_loss: {:.4f}, train_acc: {:.4f}".format(
                epoch,
                result["train_loss"],
                result["train_acc"]
            )
        )
