import tqdm
import torch
import torchvision
import dataloader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import ConfusionMatrixDisplay

import resnet


def fix_seed():
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(666)


def create_dataloader(img_size):
    train_dataset = dataloader.RetinopathyDataset(
            './data', 'train', img_size)
    train_dl = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=8,
            batch_size=64, sampler=train_dataset.wts_sampler())
    val_dataset = dataloader.RetinopathyDataset(
            './data', 'val', img_size)
    val_dl = torch.utils.data.DataLoader(
            dataset=val_dataset, num_workers=4,
            batch_size=1, sampler=val_dataset.wts_sampler())

    return train_dl, val_dl


def train_val(model, crtrn, optmzr, skdlr, device,
              train_dl, val_dl, max_epoch):
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
            running_acc = running_correct.double() / running_size
            running_loss += loss * inps.size(0)
            pbar.set_postfix(
                    acc=f"{running_acc:.2%}",
                    loss=f"{running_loss/running_size:.4f}",
                    lr=f"{current_lr[0]:.2e}" if phase == 'train' else 'None')
        return preds_list, gts_list, running_acc
    model.to(device)
    t_acc_rec, v_acc_rec = list(), list()

    best_acc = 0
    for e in range(1, max_epoch+1):
        print(f"[ INFO ] No.{e} epoch:")
        model.train()
        _, _, t_acc = inference(train_dl, phase='train')
        model.eval()
        p_list, g_list, v_acc = inference(val_dl, phase='val')
        show_cmtx(p_list, g_list, show_htmap=True if e == max_epoch else False)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), './ckpt/best.pt')
        t_acc_rec.append(t_acc.detach().cpu().numpy())
        v_acc_rec.append(v_acc.detach().cpu().numpy())

    return {'train': t_acc_rec, 'val': v_acc_rec}


def eval_model(data_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./ckpt/best82.pt'))
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    train_dl, val_dl = create_dataloader()

    model.eval()
    pbar = tqdm.tqdm(val_dl)
    running_loss, running_correct, running_size = 0, 0, 0
    gts_list, preds_list = list(), list()
    for inps, gts in pbar:
        with torch.set_grad_enabled(False):
            inps, gts = inps.to(device), gts.to(device)
            otpts = model(inps)
            loss = criterion(otpts, gts)
            preds = torch.argmax(otpts, dim=1)

            gts_list.append(gts)
            preds_list.append(preds)

        # Calculate loss & accuracy
        running_size += inps.size(0)
        running_correct += torch.sum(preds == gts)
        running_acc = running_correct.double() / running_size
        running_loss += loss * inps.size(0)
        pbar.set_postfix(
                acc=f"{running_acc:.2%}",
                loss=f"{running_loss/running_size:.4f}")
    show_cmtx(preds_list, gts_list, show_htmap=True)


def show_cmtx(preds: list, gts: list, show_htmap: bool = False):
    cmtx = np.zeros((5, 5), dtype=np.float32)
    for g, p in zip(gts, preds):
        cmtx[g, p] += 1
    num_gts = np.array([[n] for n in np.unique(gts, return_counts=True)[1]])
    cmtx /= num_gts
    if show_htmap:
        disp = ConfusionMatrixDisplay(confusion_matrix=cmtx,
                                      display_labels=list(range(5)))
        disp = disp.plot()
        plt.savefig('confusion_matrix.jpg')
    print('g/p '+''.join('{:^10}'.format('cls'+str(i)) for i in range(5)))
    print('cls0'+''.join('{:^10.2f}'.format(cmtx[0, idx]) for idx in range(5)))
    print('cls1'+''.join('{:^10.2f}'.format(cmtx[1, idx]) for idx in range(5)))
    print('cls2'+''.join('{:^10.2f}'.format(cmtx[2, idx]) for idx in range(5)))
    print('cls3'+''.join('{:^10.2f}'.format(cmtx[3, idx]) for idx in range(5)))
    print('cls4'+''.join('{:^10.2f}'.format(cmtx[4, idx]) for idx in range(5)))


def iter_train_model(max_epoch):
    model_dict = {'18_w/o_pret': resnet.resnet18(),
                  '50_w/o_pret': resnet.resnet50(),
                  '18_w/_pret': torchvision.models.resnet18(pretrained=True),
                  '50_w/_pret': torchvision.models.resnet50(pretrained=True)}
    acc_recorder = dict()
    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    train_dl, val_dl = create_dataloader((224, 224))
    # Load model
    for name, model in model_dict.items():
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 5)
        model = torch.nn.DataParallel(model)
        # Define loss
        criterion = torch.nn.CrossEntropyLoss()
        # Define optimizer
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=1e-3, momentum=0.9)
        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3e+3, gamma=0.7)

        acc_recorder[name] = train_val(model, criterion, optimizer,
                                       scheduler, device, train_dl,
                                       val_dl, max_epoch=max_epoch)

    return acc_recorder


def main(params):
    if params['mode'] == 'eval':
        train_dl, val_dl = create_dataloader((512, 512))
        eval_model(val_dl)
    else:
        recorder = iter_train_model(params['epochs'])
        pd.DataFrame(recorder).to_csv('./train_log.csv')


def param_loader():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'eval'],
                        help='Please select a mode, train or eval')
    parser.add_argument('--epochs', type=int, default=10,
                        help='How many training epochs')
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    fix_seed()
    p = param_loader()
    main(p)
