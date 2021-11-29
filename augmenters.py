from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ChannelDropout,
    InvertImg, ChannelShuffle, Cutout, MultiplicativeNoise, JpegCompression, NoOp, ToGray
)


def train_augmentations(p=0.25):
    return Compose([
        GaussNoise(var_limit=(10, 80), p=0.4),
        OneOf([
            MotionBlur(blur_limit=7, p=0.4),
            MedianBlur(blur_limit=7, p=0.4),
            Blur(blur_limit=7, p=0.4),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.09, scale_limit=0.2, rotate_limit=8),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.3),
            IAAPiecewiseAffine(p=0.3, nb_rows=2, nb_cols=2),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=3),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(brightness_limit=(-0.6, 0.7), contrast_limit=(-0.6, 0.6)),
        ], p=0.25),
        OneOf([
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.4),
            HueSaturationValue(p=0.4),
            InvertImg(p=0.2),
            ChannelShuffle(p=0.5)
        ], p=0.25),
        OneOf([
            MultiplicativeNoise(multiplier=[0.8, 1.2], per_channel=True, p=0.4),
            MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=0.4),
            MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, per_channel=True, p=0.4),
        ], p=0.25),
        JpegCompression(quality_lower=60, quality_upper=90, p=0.4)
    ], p=p)


def val_augmentations(p=1.):
    return NoOp()
