import torch
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


# Compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output,
                         weights=weights, smoothing_function=cc.method1)


def compute_batch_bleu(preds, gts):
    preds, gts = idx2char(preds), idx2char(gts)
    return np.sum([compute_bleu(p, g) for p, g in zip(preds, gts)])


# ============================================================================
# example input of Gaussian_score
#
# words = [['consult', 'consults', 'consulting', 'consulted'],
# ['plead', 'pleads', 'pleading', 'pleaded'],
# ['explain', 'explains', 'explaining', 'explained'],
# ['amuse', 'amuses', 'amusing', 'amused'], ....]
#
# the order should be : simple present, third person, present progressive, past
# ============================================================================
def gaussian_score(words):
    words_list = []
    score = 0
    yourpath = "./dataset/train.txt"
    with open(yourpath, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)


def gen_gaussian_latent(pair_num, device):
    max_len = 15    # Seqence length
    vocab_len = 29  # Latent.shape[-1]
    normal_latents = torch.empty([max_len, pair_num, vocab_len])
    normal_latents = normal_latents.normal_(mean=0, std=1)
    normal_latents = torch.repeat_interleave(normal_latents, 4, dim=1)
    tenses = torch.tensor([[i] for i in range(4)]).repeat(pair_num, 1)

    return normal_latents.to(device), tenses.to(device)


def gaussian_post_processing(preds):
    words = idx2char(preds)
    assert len(words) % 4 == 0, \
        f"Number of words should be the multiple of 4, while get {len(words)}"
    return [words[i:i+4] for i in range(int(len(words)/4))]


def char2idx(word, max_len):
    # 0: pad, 1: sos, 2: eos, 3: 'a'
    pad_num = max_len - len(word)
    idxs = [1] + [ord(c)-94 for c in word.lower()] + [2] + [0] * pad_num
    for i in idxs:
        if i > 28:
            raise ValueError

    return np.array(idxs)


def idx2char(tensor):
    array = tensor.T.detach().cpu().numpy()
    words = list()
    for a in array:
        word = ''
        for i in range(len(a)):
            # Skip idx 0
            word += chr(int(a[i])+94)
            if a[i] == 2:
                break
        words.append(word[1:-1])

    return words


def create_test_data(device):
    max_length = 15
    test_data = pd.read_table('./dataset/test.txt', delimiter=' ',
                              names=['input', 'target', 'i_tense', 't_tense'])
    input_words = torch.LongTensor([char2idx(w, max_length)
                                    for w in test_data['input'].to_list()])
    target_words = torch.LongTensor([char2idx(w, max_length)
                                    for w in test_data['target'].to_list()])
    input_tenses = torch.LongTensor([[['sp', 'tp', 'pg', 'p'].index(t)]
                                    for t in test_data['i_tense'].to_list()])
    target_tenses = torch.LongTensor([[['sp', 'tp', 'pg', 'p'].index(t)]
                                     for t in test_data['t_tense'].to_list()])

    # Seq_len first
    return input_words.T.to(device), target_words.T.to(device),\
        input_tenses.to(device), target_tenses.to(device)
