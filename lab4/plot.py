import torch
import matplotlib.pyplot as plt


def draw_training_figure(ce, kld, bleu):
    plt.figure()
    plt.title('Training Figure')
    epochs = list(range(len(ce)))
    plt.plot(epochs, ce, label='Cross Entropy')
    plt.plot(epochs, kld, label='KL Divergence')
    plt.plot(epochs, bleu, label='BLEU Score')
    plt.legend(loc='best')
    plt.show()


def draw_kl_figure(mono_kl, cyc_kl, mono_bleu, cyc_bleu):
    plt.figure()
    plt.title('KL Weight Figure')
    mono_epochs = list(range(len(mono_kl)))
    cyc_epochs = list(range(len(cyc_kl)))
    plt.plot(mono_epochs, mono_kl, label='Monotonic KLD')
    plt.plot(cyc_epochs, cyc_kl, label='Cyclical KLD')
    plt.plot(mono_epochs, mono_bleu, label='Monotonic BLEU Score')
    plt.plot(cyc_epochs, cyc_bleu, label='Cyclical BLEU Score')
    plt.legend(loc='best')
    plt.show()
    pass


def main():
    wts_path = "./0.84_0.00.pt"
    wts = torch.load(wts_path)
    print(wts.keys())
    cyc_ce = wts['ce']
    cyc_kl = wts['kld']
    cyc_bleu = wts['bleu']

    # draw_training_figure(cyc_ce, cyc_kl, cyc_bleu)

    mono_wts = torch.load('./mono_0.86_0.01.pt')
    mono_kl = mono_wts['kld']
    mono_bleu = mono_wts['bleu']
    draw_kl_figure(mono_kl, cyc_kl, mono_bleu, cyc_bleu)


if __name__ == "__main__":
    main()
