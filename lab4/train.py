import torch
import random
import tqdm
import numpy as np

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
    max_length = max(len(w) for w in words)
    words = np.array([utils.char2idx(w, max_length) for w in words],
                     dtype=np.long)

    return torch.from_numpy(words).T, torch.from_numpy(tenses)


def model_train(encoder, decoder, dataloader, optimizer,
                scheduler, device, tf_ratio=0.5):
    encoder.train(), decoder.train()
    # Training
    running_tot_size, running_loss, running_correct = 0, 0.0, 0
    running_bs = 0
    running_bleu_score = 0.0
    pbar = tqdm.tqdm(dataloader)
    for word, tense in pbar:
        word, tense = word.to(device), tense.to(device)
        # Encoder
        latent, hidden, kld_loss = encoder(e_input=word, cond=tense)
        # Decoder
        use_tf = True if random.random() < tf_ratio else False
        output, _, bce_loss = decoder(d_input=latent, cond=tense,
                                      e_input=word, tf_flag=use_tf)
        preds = torch.argmax(output, dim=-1)

        # Update
        tot_loss = bce_loss + kld_loss
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        scheduler.step()
        encoder.aneal_step_update()

        # Calculate score
        running_bleu_score += utils.compute_batch_bleu(preds, word)
        running_tot_size += word.shape[0] * word.shape[1]
        running_bs += word.shape[1]
        running_loss += tot_loss * word.shape[0] * word.shape[1]
        running_correct += torch.sum(preds == word)
        running_acc = running_correct.double() / running_tot_size

        pbar.set_postfix(loss=f"{running_loss/running_tot_size:.2f}",
                         acc=f"{running_acc:.2%}",
                         bleu_score=f"{running_bleu_score/running_bs:.2%}",
                         ratio=f"{encoder.kl_cost_annealing():.2f}")

    print(utils.idx2char(preds))
    print(utils.idx2char(word))
    return running_acc


def eval_bleu(encoder, decoder, device):
    encoder.eval(), decoder.eval()
    # Get input & target data
    i_words, t_words, i_tenses, t_tenses = utils.create_test_data(device)
    print(i_tenses)
    print(t_tenses)
    with torch.no_grad():
        latents, _, _ = encoder(i_words, i_tenses)
        outputs, _, _ = decoder(latents, t_tenses, i_words)
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
    print(pred_word_pairs[0])
    print(f"Gaussian score is {score:.2%}")

    return score


def main():
    batch_size = 16
    vocab_len = 29
    hidden_size = 128
    tf_ratio = 0.5
    max_epoch = 100
    lr = 5e-2
    kl_anneal_mode = 'mono'
    pair_num = 100
    anneal_period = 700
    num_layers = 1
    bidirectional = True
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
    decoder = model.VAE_Decoder(hidden_size, vocab_len, vocab_len,
                                num_layers, bidirectional).to(device)

    optimizer = torch.optim.SGD(list(encoder.parameters()) +
                                list(decoder.parameters()),
                                lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, anneal_period, 0.5)
    for e in range(max_epoch):
        print()
        print(f"[ INFO ] No.{e} epoch")
        _ = model_train(encoder, decoder, dataloader, optimizer, scheduler,
                        device, tf_ratio=tf_ratio)
        eval_acc, bleu_acc = eval_bleu(encoder, decoder, device)
        gaussian_score = eval_gaussian(decoder, device, pair_num)
        print(f"Learning rate now is {scheduler.get_last_lr()}")
    torch.save(encoder.state_dict(), './ckpt/encoder.pt')
    torch.save(decoder.state_dict(), './ckpt/decoder.pt')


if __name__ == "__main__":
    main()
