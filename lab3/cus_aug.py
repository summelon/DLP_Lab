import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import PIL


class ImgAugTransform:
    def __init__(self, mode):
        def sometimes(aug):
            return iaa.Sometimes(0.5, aug)
        if mode == 'train':
            self.aug = iaa.Sequential([
                iaa.Resize((224, 224)),
                iaa.Fliplr(0.5),
                sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                        cval=(0, 255),
                        mode=ia.ALL)),
                iaa.SomeOf((0, 5), [
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                        iaa.Invert(0.05, per_channel=True),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.AddToHueAndSaturation((-20, 20)),
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        sometimes(iaa.ElasticTransformation(
                            alpha=(0.5, 3.5), sigma=0.25)),
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ], random_order=True)
                ])
        else:
            self.aug = iaa.Sequential([iaa.Resize((224, 224))])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = PIL.Image.fromarray(img)

        return img

    def check_aug(self, imgs):
        self.aug.show_grid(imgs, cols=8, rows=8)
