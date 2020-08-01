import tqdm
import torch
import torchvision
import dataloader
import numpy as np


def fix_seed():
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(666)


def create_dataloader():
    train_dataset = dataloader.RetinopathyDataset('./data', 'train')
    train_dl = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=8, shuffle=True, batch_size=64)
    val_dataset = dataloader.RetinopathyDataset('./data', 'val')
    val_dl = torch.utils.data.DataLoader(
            dataset=val_dataset, num_workers=4, shuffle=False, batch_size=1)

    return train_dl, val_dl


def train_val(model, crtrn, optmzr, skdlr, device):
    def inference(dl, phase):
        pbar = tqdm.tqdm(dl)
        running_loss, running_correct, running_size = 0, 0, 0
        gts_list, preds_list = list(), list()
        for inps, gts in pbar:
            with torch.set_grad_enabled(phase == 'train'):
                inps, gts = inps.to(device), gts.to(device)
                otpts = model(inps)
                loss = crtrn(otpts, gts)
                preds = torch.argmax(otpts, dim=1)

            if phase == 'train':
                optmzr.zero_grad()
                loss.backward()
                optmzr.step()
                skdlr.step()
                current_lr = skdlr.get_last_lr()
            else:
                gts_list.append(gts)
                preds_list.append(preds)

            # Calculate loss & accuracy
            running_size += inps.size(0)
            running_correct += torch.sum(preds == gts)
            running_loss += loss * inps.size(0)
            pbar.set_postfix(
                    loss=f"{running_loss/running_size:.4f}",
                    acc=f"{running_correct.double()/running_size:.2%}",
                    lr=f"{current_lr[0]:.2e}" if phase == 'train' else 'None')
        return preds_list, gts_list
    model.to(device)
    train_dl, val_dl = create_dataloader()
    epochs = 10

    for e in range(1, epochs+1):
        print(f"[ INFO ] No.{e} epoch:")
        model.train()
        _, _ = inference(train_dl, phase='train')
        model.eval()
        p_list, g_list = inference(val_dl, phase='val')
        show_cmtx(p_list, g_list, show_htmap=True if e == epochs else False)


def show_cmtx(preds: list, gts: list, show_htmap: bool = False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    cmtx = np.zeros((5, 5), dtype=np.int64)
    for g, p in zip(gts, preds):
        cmtx[g, p] += 1
    if show_htmap:
        disp = ConfusionMatrixDisplay(confusion_matrix=cmtx,
                                      display_labels=list(range(5)))
        disp = disp.plot()
        plt.savefig('confusion_matrix.jpg')
    print(' g/p  '+''.join('{:^10}'.format('cls'+str(i)) for i in range(5)))
    print(' cls0 '+''.join('{:^10d}'.format(cmtx[0, idx]) for idx in range(5)))
    print(' cls1 '+''.join('{:^10d}'.format(cmtx[1, idx]) for idx in range(5)))
    print(' cls2 '+''.join('{:^10d}'.format(cmtx[2, idx]) for idx in range(5)))
    print(' cls3 '+''.join('{:^10d}'.format(cmtx[3, idx]) for idx in range(5)))
    print(' cls4 '+''.join('{:^10d}'.format(cmtx[4, idx]) for idx in range(5)))


def main():
    # Load model
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3, momentum=0.9, weight_decay=5e-4)
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5e+2, gamma=0.9)
    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    train_val(model, criterion, optimizer, scheduler, device)


if __name__ == "__main__":
    fix_seed()
    main()
