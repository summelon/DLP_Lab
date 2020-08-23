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
    z = torch.randn((len(labels), latent_dim)).to(labels.device)
    gen_img = generator(z, labels)

    return gen_img


def main():
    generator_weight_path = "./ckpt/acgan/acgan.pth"
    classifer_weight_path = "./ckpt/classifier_weight.pth"
    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = acgan.Generator(img_size=64, latent_dim=100).to(device)
    saved_dict = torch.load(generator_weight_path)
    generator.load_state_dict(saved_dict['generator'])
    classifier = evaluator.evaluation_model(classifer_weight_path)
    generator.eval(), classifier.resnet18.eval()

    dataset = SyntheticDataset(img_size=666, mode='val')
    labels = torch.cat([data[1].view(1, -1) for data in list(dataset)])
    labels = labels.type(torch.FloatTensor)

    imgs = gen_sample(generator, labels.to(device))
    acc = classifier.eval(imgs, labels)
    print(acc)


if __name__ == '__main__':
    main()
