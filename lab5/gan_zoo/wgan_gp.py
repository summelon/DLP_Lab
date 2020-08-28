import os
import sys
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image


sys.path.append('/home/shortcake/project/DLP_Lab/lab5')
import dataset
import eval_gen

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def params_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use "
                             "during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=24,
                        help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval between image sampling")
    parser.add_argument("--noise_ratio", type=float, default=0.15,
                        help="Gaussian noise add into input of discriminator")
    parser.add_argument("--noise_decay", type=float, default=0.99,
                        help="Decay per epoch for gaussian noise add to discriminator")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of training steps for discriminator per iter")
    opt = parser.parse_args()
    print(opt)

    return opt


def fix_seed():
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(666)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = [torch.randn((1, opt.latent_dim)) for i in range(n_row ** 2)]
    z = torch.cat(z, dim=0).type(FloatTensor)
    labels = dataset.gen_labels(opt.n_classes, n_row ** 2).type(FloatTensor)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/wgan_gp/%d.png" % batches_done, nrow=n_row, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, real_labels, fake_labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    import torch.autograd as autograd

    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    alpha = alpha.view(-1, 1)
    inter_labels = (alpha * real_labels + ((1 - alpha) * fake_labels)).requires_grad_(True)
    d_interpolates = D(interpolates, inter_labels).view(-1, 1)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Generator, self).__init__()

        n_classes = 24
        nc = 3
        ngf = img_size
        self.fc = nn.Sequential(
                nn.Linear(latent_dim+n_classes, 256, bias=True),
                nn.ReLU(True),
                nn.Linear(256, latent_dim, bias=True))

        self.conv_block = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=True),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
                nn.Tanh())

    def forward(self, noise, labels):
        cond_noise = torch.cat([noise, labels], dim=-1)
        cond_noise = self.fc(cond_noise)
        cond_noise = cond_noise.view(noise.size(0), noise.size(1), 1, 1)
        img = self.conv_block(cond_noise)

        return img


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        n_classes = 24
        ndf = img_size
        nc = 3
        img_shape = (nc, img_size, img_size)
        input_size = np.prod(img_shape)
        self.transform = nn.Sequential(
                nn.Linear(input_size+n_classes, 1024, bias=True),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, input_size, bias=True))

        self.conv_block = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True))

    def forward(self, img, cond):
        cond_img = torch.cat([img.view(img.size(0), -1), cond], dim=-1)
        cond_img = self.transform(cond_img).view(img.shape)
        validity = self.conv_block(cond_img)
        return validity


if __name__ == "__main__":
    os.makedirs("images/wgan_gp", exist_ok=True)
    os.makedirs("ckpt/wgan_gp", exist_ok=True)
    fix_seed()

    opt = params_loader()

    # Initialize generator and discriminator
    generator = Generator(opt.img_size, opt.latent_dim)
    discriminator = Discriminator(opt.img_size)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
                    dataset.SyntheticDataset(opt.img_size),
                    batch_size=opt.batch_size, shuffle=True, num_workers=8)

    # Optimizers
    optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    test_noise = [torch.randn(1, opt.latent_dim) for i in range(32)]
    test_noise = torch.cat(test_noise, dim=0).type(FloatTensor)
    best_acc = 0
    noise_ratio = opt.noise_ratio
    for epoch in range(opt.n_epochs):
        run_g_loss, run_d_loss, run_size = 0, 0, 0
        run_correct, run_gt = 0, 0
        print()
        print(f"Epoch {epoch}:")
        pbar = tqdm.tqdm(dataloader)
        for i, (imgs, labels) in enumerate(pbar):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(
                    FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(
                    FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (len(imgs), opt.latent_dim))))
            # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            gen_labels = dataset.gen_labels(opt.n_classes, len(imgs)).type(FloatTensor)

            # Generate a batch of images
            gen_imgs = generator(z, labels)

            if i % opt.n_critic == 0:
                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_imgs, labels)
                g_adv_loss = -torch.mean(validity)
                g_loss = g_adv_loss

                g_loss.backward()
                optimizer_G.step()

                run_g_loss += g_loss.double() * len(imgs)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Add gaussian noise to real image
            # noise = torch.cat([torch.randn(imgs.shape[1:]).unsqueeze(0) for i in range(imgs.shape[0])], dim=0).to(real_imgs.device)
            # real_imgs = real_imgs * (1 - noise_ratio) + noise * noise_ratio

            # Loss for real images
            real_pred = discriminator(real_imgs, labels)
            # Loss for fake images
            if torch.rand(1) > 0.5:
                fake_imgs = real_imgs
                fake_labels = gen_labels
            else:
                fake_imgs = gen_imgs.detach()
                fake_labels = labels
            fake_pred = discriminator(fake_imgs, fake_labels)

            lambda_gp = 10
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels, fake_labels)
            d_adv_loss = -(torch.mean(real_pred) - torch.mean(fake_pred)) + lambda_gp * gradient_penalty
            d_loss = d_adv_loss

            d_loss.backward()
            optimizer_D.step()

            run_d_loss += d_loss.double() * len(imgs)
            run_size += len(imgs)

            pbar.set_postfix(D_loss=f'{run_d_loss.item()/run_size:.3f}',
                             G_loss=f'{run_g_loss.item()/(run_size/opt.n_critic):.3f}')
                             # ratio=f'{noise_ratio:.3%}')
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

        # Ratio decay
        # if run_acc > 0.60:
        #     noise_ratio *= opt.noise_decay

        # Test on evaluator from TA
        generator.eval()
        test_acc = eval_gen.eval_gen(generator, test_noise)
        generator.train()
        print(f"Test accuracy on evaluator is {test_acc:.2%}")
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save({'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'acc': test_acc},
                       'ckpt/wgan_gp/wgan_gp.pth')
