import torch
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters())
    # set up one cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):

        # Training phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)

            # calculates gradients
            loss.backward()

            # perform gradient descent and modifies the weights
            optimizer.step()

            # reset the gradients
            optimizer.zero_grad()

            # record and update lr
            lrs.append(get_lr(optimizer))

            # modifies the lr value
            sched.step()

        # Evaluation phase (after all weight updates in this epoch)
        # val_result = evaluate(model, val_loader)
        result = {
            # 'val_loss': val_result['loss'],
            # 'val_acc': val_result['acc'],
            'train_acc': evaluate_train(model, train_loader),
            'train_loss': torch.stack(train_losses).mean().item(),
            'lrs': lrs
        }

        model.epoch_end(epoch, result)
        history.append(result)

    return history


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    outputs = [model.eval_step(batch) for batch in dl]
    return model.eval_epoch_end(outputs)


@torch.no_grad()
def evaluate_train(model, train_loader):
    model.eval()
    outputs = [model.train_acc_step(batch) for batch in train_loader]
    return model.train_acc_epoch_end(outputs)
