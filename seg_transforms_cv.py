import numpy as np
from skimage import img_as_float
from seg_transforms import SegTransform

class SegCVTransformNormalizeToTensor (SegTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def transform_single(self, sample):
        sample = sample.copy()

        # Convert image to float
        image = img_as_float(sample['image_arr'])

        if image.shape[2] == 4:
            # Has alpha channel introduced by padding
            # Split the image into RGB/alpha
            alpha_channel = image[:, :, 3:4]
            image = image[:, :, :3]

            # Account for the alpha during standardisation
            if self.mean is not None and self.std is not None:
                image = (image - (self.mean[None, None, :] * alpha_channel)) / self.std[None, None, :]
        else:
            # Standardisation
            if self.mean is not None and self.std is not None:
                image = (image - self.mean[None, None, :]) / self.std[None, None, :]

        # Convert to NCHW tensors
        assert image.shape[2] == 3
        sample['image'] = image.transpose(2, 0, 1).astype(np.float32)
        del sample['image_arr']
        if 'labels_arr' in sample:
            sample['labels'] = sample['labels_arr'][None, ...].astype(np.int64)
            del sample['labels_arr']
        if 'mask_arr' in sample:
            sample['mask'] = img_as_float(sample['mask_arr'])[None, ...].astype(np.float32)
            del sample['mask_arr']

        return sample
  