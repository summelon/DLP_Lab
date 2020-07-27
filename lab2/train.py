import torch
import dataloader
import EEGNet


def train_val(model, criterion, optimizer, max_epoch,
              t_data, t_lbl, v_data, v_lbl):
    for epoch in range(1, max_epoch+1):
        # Training
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            train_outputs, train_acc = inference(t_data, t_lbl)
            loss = criterion(train_outputs, t_lbl)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        _, val_acc = inference(v_data, v_lbl)
        print(f"[{epoch:^5d}]Train: {train_acc:.2%}, Val: {val_acc:.2%}",
              end="\r" if epoch != max_epoch else '')


# Start training
def inference(model, inps, lbls):
    otpts = model(inps)
    preds = torch.argmax(otpts, dim=1)
    acc = torch.sum(preds == lbls).item() / len(inps)

    return otpts, acc


def main():
    # Load data
    t_data, t_lbl, v_data, v_lbl = dataloader.read_bci_data('./dataset')
    # Load model
    model = EEGNet.EEGNet(activation=torch.nn.LeakyReLU)
    model.to('cuda:0')
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    max_epoch = 300
    train_val(model, criterion, optimizer, max_epoch,
              t_data, t_lbl, v_data, v_lbl)


if __name__ == "__main__":
    main()
