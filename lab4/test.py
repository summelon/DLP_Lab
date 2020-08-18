import torch
import random
import tqdm
import numpy as np
from argparse import ArgumentParser

import model
import dataset
import utils


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def eval_bleu(encoder, decoder, device):
    encoder.eval(), decoder.eval()
    # Get input & target data
    i_words, t_words, i_tenses, t_tenses = utils.create_test_data(device)
    with torch.no_grad():
        latents, _, _ = encoder(i_words, i_tenses)
        outputs, _, _ = decoder(latents, t_tenses)
        preds = torch.argmax(outputs, dim=-1)
        accuracy = torch.sum(preds == t_words).double() / t_words.numel()
        bleu_acc = utils.compute_batch_bleu(preds, t_words) / t_words.shape[1]
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
    # for i in range(10):
    #     print(pred_word_pairs[i])
    print(f"Gaussian score is {score:.2f}")

    return score


def main(params):
    pair_num = 100
    vocab_len = 29
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    kl_anneal_mode = params['anneal_mode']
    anneal_period = params['anneal_period'] / batch_size
    num_layers = params['num_layers']
    bidirectional = params['bidirection']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fix seed by given number
    fix_seed(params['seed'])
    # Create model: encoder & decoder
    encoder = model.VAE_Encoder(vocab_len, hidden_size, vocab_len,
                                num_layers, bidirectional,
                                kl_anneal_mode, anneal_period).to(device)
    decoder = model.VAE_Decoder(hidden_size, vocab_len, params['dropout_rate'],
                                num_layers, bidirectional).to(device)
    weights = torch.load(params['wts_path'])
    encoder.load_state_dict(weights['encoder'])
    decoder.load_state_dict(weights['decoder'])

    eval_acc, bleu_score = eval_bleu(encoder, decoder, device)
    gaussian_score = eval_gaussian(decoder, device, pair_num)


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
    parser.add_argument("--wts_path", type=str,
                        help="Path to trained weights")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed search")
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
