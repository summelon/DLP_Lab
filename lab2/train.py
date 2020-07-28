import sys
import torch
import dataloader
import EEGNet
import DeepConvNet
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def inference(m, inps, lbls):
    otpts = m(inps)
    preds = torch.argmax(otpts, dim=1)
    acc = torch.sum(preds == lbls).item() / len(inps)

    return otpts, acc


def train_val(model, criterion, optimizer, max_epoch,
              t_data, t_lbl, v_data, v_lbl):
    acc_rec = {'train': [], 'val': []}
    best_acc = 0
    for epoch in range(1, max_epoch+1):
        # Training
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            train_outputs, train_acc = inference(model, t_data, t_lbl)
            loss = criterion(train_outputs, t_lbl)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        _, val_acc = inference(model, v_data, v_lbl)
        acc_rec['train'].append(train_acc)
        acc_rec['val'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       './weights/'+model.name+model.act_name+'.pt')
        print(f"[{epoch:^5d}]Train_acc: {train_acc:.2%}, "
              f"Train_loss: {loss:.2f}, best val_acc: {best_acc:.2%}",
              end="\r" if epoch != max_epoch else '\n')

    return acc_rec


def draw_plot(m_name, act_rec):
    plt.figure()
    plt.title(f"Activation Function Comparison({m_name})")
    colors = plt.get_cmap('tab20c')(np.linspace(0, 1, 6))
    counter = 0
    for act_name, rec in act_rec.items():
        for t_or_v, log in rec.items():
            plt.plot(range(len(log)), log,
                     c=colors[counter], label=act_name+'_'+t_or_v)
            counter += 1
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(axis='both')
    plt.show()


def iter_test_act(model_func):
    act_map = {'Relu': torch.nn.ReLU,
               'Leaky_relu': torch.nn.LeakyReLU,
               'Elu': torch.nn.ELU}
    max_epoch = 300
    # Load data
    t_data, t_lbl, v_data, v_lbl = dataloader.read_bci_data('./dataset')
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Recorder of train & val for different activation
    act_rec = dict()
    # Load model
    for name, act in act_map.items():
        print('-'*77)
        print(f"[ INFO ]Testing activation function: {name}")
        model = model_func(activation=act)
        model.to('cuda:0')
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        act_rec[name] = train_val(model, criterion, optimizer, max_epoch,
                                  t_data, t_lbl, v_data, v_lbl)

    return act_rec


def eval_model(model_func):
    act_map = {'Relu': torch.nn.ReLU,
               'Leaky_relu': torch.nn.LeakyReLU,
               'Elu': torch.nn.ELU}
    # Load data
    t_data, t_lbl, v_data, v_lbl = dataloader.read_bci_data('./dataset')
    # Load model
    for name, act in act_map.items():
        model = model_func(activation=act)
        try:
            weight = torch.load('./weights/'+model.name+model.act_name+'.pt')
        except FileNotFoundError:
            print(f'You have not saved weights for {model.name}')
            sys.exit()
        model.load_state_dict(weight)
        model.to('cuda:0')
        # Test
        model.eval()
        _, val_acc = inference(model, v_data, v_lbl)
        print(f"{name}: validation accuracy: {val_acc:.2%}")


def main(params):
    if params['model_name'] == 'eegnet':
        model_function = EEGNet.EEGNet
    elif params['model_name'] == 'deepconvnet':
        model_function = DeepConvNet.DeepConvNet
    else:
        raise ValueError(f"{params['model_name']} is not a correct name.")
    print(f"[ INFO ]Using {params['model_name']} as model")
    if params['mode'] == 'train':
        activation_recorder = iter_test_act(model_function)
        draw_plot(params['model_name'], activation_recorder)
    elif params['mode'] == 'val':
        eval_model(model_function)
    else:
        raise ValueError(f"Mode name {params['mode']} is wrong")


def param_loader():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='eegnet',
                        choices=['eegnet', 'deepconvnet'],
                        help='Please select a model.')
    parser.add_argument('--mode', type=str, default='val',
                        choices=['train', 'val'], help='Please select mode.')
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
