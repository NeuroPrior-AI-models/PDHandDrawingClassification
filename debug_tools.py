from train import evaluate, evaluate_train


def try_batch(model, dl):
    for imgs, labels in dl:
        print("images shape : ", imgs.shape)
        print("labels : ", labels)
        outs = model(imgs)
        print("outs.shape :", outs.shape)
        print("outs : ", outs)
        break


def check_eval(model, train_dl, val_dl):
    print(evaluate(model, val_dl))
    print(evaluate_train(model, train_dl))
