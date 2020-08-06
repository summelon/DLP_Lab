import pandas as pd
import matplotlib.pyplot as plt

data_dict = {
        '18_w/o_pret': {
            'train': [0.73144952, 0.73493719, 0.73486601, 0.73493719,
                      0.73518631, 0.73493719, 0.73486601, 0.73515072,
                      0.73518631, 0.73511513],
            'val': [0.73352313, 0.73338078, 0.73338078, 0.73437722,
                    0.73181495, 0.73380783, 0.73352313, 0.73323843,
                    0.73338078, 0.73366548]},
        '18_w/_pret': {
            'train': [0.74031104, 0.75774939, 0.76792768, 0.77419125, 0.78034806, 0.78479661, 0.79024165, 0.79486814, 0.79657639, 0.79842699],
            'val': [0.75245552, 0.76384342, 0.76711744, 0.77451957, 0.77935943, 0.78149466, 0.78234875, 0.78761566, 0.78761566, 0.78846975]},
        '50_w/o_pret': {
            'train': [0.73109363, 0.73287306, 0.73258835, 0.73297982, 0.73276629, 0.73255276, 0.733051, 0.73330012, 0.73347806, 0.73337129],
            'val': [0.73352313, 0.73352313, 0.73352313, 0.73352313, 0.73352313, 0.73352313, 0.72967972, 0.73323843, 0.73352313, 0.72996441]},
        '50_w/_pret': {
            'train': [0.73995516, 0.76949358, 0.78138012, 0.79049076, 0.7977864, 0.80184348, 0.80686145, 0.81362326, 0.81821417, 0.8239795],
            'val': [0.75629893, 0.77010676, 0.78021352, 0.78718861, 0.79131673, 0.793879, 0.79601423, 0.79430605, 0.79914591, 0.79814947]}}


def draw_configure_18():
    epoch_len = len(data_dict['18_w/o_pret']['train'])
    plt.figure()
    plt.title('Result Comparison(ResNet18)')
    plt.plot(range(epoch_len), data_dict['18_w/o_pret']['train'], label='train_w/o_pretrain')
    plt.plot(range(epoch_len), data_dict['18_w/o_pret']['val'], label='val_w/o_pretrain')
    plt.plot(range(epoch_len), data_dict['18_w/_pret']['train'], label='train_w/_pretrain')
    plt.plot(range(epoch_len), data_dict['18_w/_pret']['val'], label='val_w/_pretrain')
    plt.legend(loc='best')

    plt.savefig('compare18.jpg')


def draw_configure_50():
    epoch_len = len(data_dict['50_w/o_pret']['train'])
    plt.figure()
    plt.title('Result Comparison(ResNet50)')
    plt.plot(range(epoch_len), data_dict['50_w/o_pret']['train'], label='train_w/o_pretrain')
    plt.plot(range(epoch_len), data_dict['50_w/o_pret']['val'], label='val_w/o_pretrain')
    plt.plot(range(epoch_len), data_dict['50_w/_pret']['train'], label='train_w/_pretrain')
    plt.plot(range(epoch_len), data_dict['50_w/_pret']['val'], label='val_w/_pretrain')
    plt.legend(loc='best')

    plt.savefig('compare50.jpg')

if __name__ == "__main__":
    draw_configure_18()
    draw_configure_50()
