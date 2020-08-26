import sys
import torch
import numpy as np

sys.path.append('/home/shortcake/project/DLP_Lab/lab5')
import acgan
import evaluator
from dataset import SyntheticDataset


def fix_seed():
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(666)


def gen_sample(generator, labels):
    latent_dim = 100
    z = [torch.randn(1, latent_dim) for i in range(len(labels))]
    z = torch.cat(z, dim=0).to(labels.device)
    gen_img = generator(z, labels)

    return gen_img


def denorm(imgs):
    min_value = float(imgs.min())
    max_value = float(imgs.max())
    # Shift to range(0, 1)
    imgs.clamp(min=min_value, max=max_value)
    imgs.add_(-min_value).div_(max_value - min_value + 1e-5)
    # Unnormalization and channel last
    imgs.mul_(255).add_(0.5).clamp(0, 255).permute(0, 2, 3, 1)
    # Normalize to (-1, 1) again and channel first
    imgs.div_(255).sub_(0.5).div_(0.5).permute(0, 3, 1, 2)

    return imgs


def eval_gen(generator, device):
    classifer_weight_path = "./ckpt/classifier_weight.pth"
    classifier = evaluator.evaluation_model(classifer_weight_path)
    classifier.resnet18.eval()

    dataset = SyntheticDataset(img_size=666, mode='val')
    labels = torch.cat([data[1].view(1, -1) for data in list(dataset)])
    labels = labels.type(torch.FloatTensor).to(device)

    imgs = gen_sample(generator, labels)
    imgs = denorm(imgs)

    return classifier.eval(imgs, labels)


def main():
    generator_weight_path = "./ckpt/acgan/acgan.pth"
    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = acgan.Generator(img_size=64, latent_dim=100).to(device)
    saved_dict = torch.load(generator_weight_path)
    generator.load_state_dict(saved_dict['generator'])
    generator.eval()

    acc = eval_gen(generator, device)
    print(acc)


if __name__ == '__main__':
    main()
