import torch
import random
import tqdm
import copy
import numpy as np
from argparse import ArgumentParser

import model
import dataset
import utils


def fix_seed():
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(666)


def text_fn(pairs):
    tenses = np.array([[t] for _, t in pairs], dtype=np.long)
    words = np.array([w for w, _ in pairs])
    # max_length = max(len(w) for w in words)
    max_length = 15
    words = np.array([utils.char2idx(w, max_length) for w in words],
                     dtype=np.long)

    return torch.from_numpy(words).T, torch.from_numpy(tenses)


def model_train(encoder, decoder, dataloader,
                e_optimizer, d_optimizer, device):
    encoder.train(), decoder.train()
    # Training
    running_tot_size, running_bs, running_correct = 0, 0, 0
    running_kld_loss, running_ce_loss = 0.0, 0.0
    running_bleu_score = 0.0
    pbar = tqdm.tqdm(dataloader)
    for word, tense in pbar:
        word, tense = word.to(device), tense.to(device)
        # Encoder
        latent, _, kld_loss = encoder(word, tense)
        # Decoder
        dynamic_ratio = max(1-encoder.kl_anneal_step/encoder.period, 0.5)
        use_tf = True if random.random() < dynamic_ratio else False
        output, _, ce_loss = decoder(latent, tense,
                                     e_input=word, tf_flag=use_tf)
        preds = torch.argmax(output, dim=-1)

        # Update
        tot_loss = ce_loss + kld_loss
        e_optimizer.zero_grad()
        d_optimizer.zero_grad()
        tot_loss.backward()
        e_optimizer.step()
        d_optimizer.step()
        encoder.aneal_step_update()

        # Calculate score
        running_bleu_score += utils.compute_batch_bleu(preds, word)
        running_tot_size += word.shape[0] * word.shape[1]
        running_bs += word.shape[1]
        # running_loss += tot_loss * word.shape[0] * word.shape[1]
        running_kld_loss += kld_loss * word.shape[0] * word.shape[1]
        running_ce_loss += ce_loss * word.shape[0] * word.shape[1]
        running_correct += torch.sum(preds == word)
        running_acc = running_correct.double() / running_tot_size

        pbar.set_postfix(kld=f"{running_kld_loss/running_tot_size:.2f}",
                         ce=f"{running_ce_loss/running_tot_size:.2f}",
                         acc=f"{running_acc:.2%}",
                         bleu_score=f"{running_bleu_score/running_bs:.2%}",
                         ratio=f"{encoder.kl_cost_annealing():.2f}")
    ce_loss = running_ce_loss/running_tot_size
    kld_loss = running_kld_loss/running_tot_size

    print(utils.idx2char(preds))
    print(utils.idx2char(word))
    return ce_loss, kld_loss


def eval_bleu(encoder, decoder, device):
    encoder.eval(), decoder.eval()
    # Get input & target data
    i_words, t_words, i_tenses, t_tenses = utils.create_test_data(device)
    with torch.no_grad():
        latents, _, _ = encoder(i_words, i_tenses)
        outputs, _, _ = decoder(latents, t_tenses)
        preds = torch.argmax(outputs, dim=-1)
        accuracy = torch.sum(preds == t_words).double() / t_words.numel()
        bleu_acc = utils.compute_batch_bleu(preds, t_words) / len(t_words)
    print(utils.idx2char(preds))
    print(utils.idx2char(t_words))
    print(f"Accuracy is {accuracy:.2%}")
    print(f"BLEU Accuracy is {bleu_acc:.2%}")

    return accuracy, bleu_acc


def eval_gaussian(decoder, device, pair_num):
    decoder.eval()
    latents, tenses = utils.gen_gaussian_latent(pair_num, device)
    outputs, _, _ = decoder(latents, tenses)
    preds = torch.argmax(outputs, dim=-1)
    pred_word_pairs = utils.gaussian_post_processing(preds)
    score = utils.gaussian_score(pred_word_pairs)
    print()
    for i in range(10):
        print(pred_word_pairs[i])
    print(f"Gaussian score is {score:.2%}")

    return score


def main(params):
    pair_num = 100
    vocab_len = 29
    max_epoch = 1000
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    lr = params['lr']
    kl_anneal_mode = params['anneal_mode']
    anneal_period = params['anneal_period'] / batch_size
    num_layers = params['num_layers']
    bidirectional = params['bidirection']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fix seed
    fix_seed()
    # Create dataloader
    ds = dataset.dataset()
    dataloader = torch.utils.data.DataLoader(
            dataset=ds, num_workers=8, batch_size=batch_size,
            shuffle=True, collate_fn=text_fn)
    # Create model: encoder & decoder
    encoder = model.VAE_Encoder(vocab_len, hidden_size, vocab_len,
                                num_layers, bidirectional,
                                kl_anneal_mode, anneal_period).to(device)
    decoder = model.VAE_Decoder(hidden_size, vocab_len, params['dropout_rate'],
                                num_layers, bidirectional).to(device)

    e_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=0.9)
    d_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr, momentum=0.9)

    best_bleu, best_gaussian = 0.0, 0.0
    best_gaussian_wts, best_bleu_wts = dict(), dict()
    kld_list, ce_list, bleu_list = list(), list(), list()
    for e in range(max_epoch):
        print()
        print(f"[ INFO ] No.{e} epoch")
        ce_loss, kld_loss = model_train(encoder, decoder, dataloader,
                                        e_optimizer, d_optimizer, device)
        eval_acc, bleu_score = eval_bleu(encoder, decoder, device)
        gaussian_score = eval_gaussian(decoder, device, pair_num)
        kld_list.append(kld_loss), ce_list.append(ce_loss)
        bleu_list.append(bleu_score)
        if best_gaussian < gaussian_score:
            best_gaussian = gaussian_score
            best_gaussian_wts = copy.deepcopy(
                    {'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict()},)
        if best_bleu < bleu_score:
            best_bleu = bleu_score
            best_bleu_wts = copy.deepcopy(
                    {'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict()},)
        if best_gaussian == 0.05:
            break
    recorder = {'kld': kld_list, 'ce': ce_list, 'bleu': bleu_list}
    if best_gaussian > 0:
        print("Save best gaussian weight!")
        torch.save(best_gaussian_wts.update(recorder),
                   f'./ckpt/{best_bleu:.2f}_{best_gaussian:.2f}.pt')
    else:
        torch.save(best_bleu_wts.update(recorder),
                   f'./ckpt/{best_bleu:.2f}_{best_gaussian:.2f}.pt')


def param_loader():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size")
    parser.add_argument("--lr", type=float, default=5e-2,
                        help="Learning rate")
    parser.add_argument("--anneal_mode", type=str,
                        choices=['mono', 'cyc'], default='cyc',
                        help="anneal model for both KL weight "
                             "and teacher forcing")
    parser.add_argument("--anneal_period", type=int, default=1e+4,
                        help="Final value = set value / batch size")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers for LSTM, "
                             "default 2 for dropout")
    parser.add_argument("--bidirection", type=bool, default=True,
                        help="Use bidirection in LSTM or not")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                        help="Dropout rate in decoder")
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
