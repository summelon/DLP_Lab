import sys
import torch
import numpy as np
from argparse import ArgumentParser

sys.path.append('/home/shortcake/project/DLP_Lab/lab5')
# import acgan
import wgan_gp
import evaluator
from dataset import SyntheticDataset


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def denorm(imgs):
    min_value = float(imgs.min())
    max_value = float(imgs.max())
    # Shift to range(0, 1)
    imgs.clamp_(min=min_value, max=max_value)
    imgs.add_(-min_value).div_(max_value - min_value + 1e-5)
    # Unnormalization and channel last
    imgs.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1)
    # Normalize to (-1, 1) again and channel first
    imgs.div_(255).sub_(0.5).div_(0.5).permute(0, 3, 1, 2)

    return imgs


def eval_gen(generator, noise):
    classifer_weight_path = "./ckpt/classifier_weight.pth"
    classifier = evaluator.evaluation_model(classifer_weight_path)
    classifier.resnet18.eval()

    dataset = SyntheticDataset(img_size=666, mode='val')
    labels = torch.cat([data[1].view(1, -1) for data in list(dataset)])
    labels = labels.type(torch.FloatTensor).to(noise.device)

    imgs = generator(noise, labels)
    imgs = denorm(imgs)

    return classifier.eval(imgs, labels)


def main(params):
    generator_weight_path = params['weight']
    fix_seed(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exec('import ' + params['model'])
    generator = eval(
            f"{params['model']}.Generator(img_size=64, latent_dim=100)")
    generator = generator.to(device)
    saved_dict = torch.load(generator_weight_path)
    generator.load_state_dict(saved_dict['generator'])
    generator.eval()

    test_noise = [torch.randn(1, 100) for i in range(32)]
    test_noise = torch.cat(test_noise, dim=0).to(device)
    acc = eval_gen(generator, test_noise)
    print(acc)


def param_loader():
    parser = ArgumentParser()
    parser.add_argument("--weight", type=str,
                        help="Path to where saved weight is")
    parser.add_argument("--model", type=str, choices=['acgan', 'wgan_gp'],
                        help="Use what kinds of GAN model")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed number")
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    p = param_loader()
    main(p)
