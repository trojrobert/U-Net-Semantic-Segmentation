
from torch.utils.data import Dataset, Sampler


class SegAccessor (Dataset):
    def __init__(self, ds, labels, mask, xf, pair, transforms, pipeline_type='cv', include_indices=False):
        """
        Generates samples.

        Can generate samples for either a Pillow (pipeline_type='pil') or OpenCV (pipeline_type='cv')
        based pipeline.

        Pillow samples take the form:
            {'image_pil': PIL.Image,                # input image
             [optional] 'labels_pil': PIL.Image,    # labels image
             [optional] 'mask_pil': PIL.Image,      # mask image
             [optional] 'xf_pil': np.array}         # transformation as NumPy array

        OpenCV samples take the form:
            {'image_arr': np.array,                 # input image as a `(H, W, C)` array
             [optional] 'labels_arr': np.array,     # labels image as a `(H, W)` array
             [optional] 'mask_arr': np.array,       # mask image as a `(H, W)` array
             [optional] 'xf_cv': np.array}          # transformation as NumPy array

        :param ds: data source to load from
        :param labels: flag indicating if the ground truth labels should be loaded
        :param mask: flag indicating if mask should be loaded
        :param xf: flag indicating if transformation should be loaded
        :param pair: flag indicating if sample should be 'paired' for standard augmentation driven consistency
            regularization.  If True, each sample will be a dict of `{'sample0': <sample>
        :param transforms: optional transformation to apply to each sample when retrieved
        :param pipeline_type: pipeline type 'pil' | 'cv'
        :param include_indices: if True, include sample index in each sample
        """
        super(SegAccessor, self).__init__()

        if pipeline_type not in {'pil', 'cv'}:
            raise ValueError('pipeline_type should be either \'pil\' or \'cv\', not {}'.format(pipeline_type))

        self.ds = ds
        self.labels_flag = labels
        self.mask_flag = mask
        self.xf_flag = xf
        self.pair_flag = pair
        self.transforms = transforms
        self.pipeline_type = pipeline_type
        self.include_indices = include_indices

class DataSource (object):
    def save_prediction_by_index(self, out_dir, pred_y_arr, sample_index):
        save_prediction(out_dir, pred_y_arr, self.sample_names[sample_index])

    def get_mean_std(self):
        # For now:
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])\

class SegCollate (object):
    def __init__(self, block_size, batch_aug_fn=None):
        self.block_size = block_size
        self.batch_aug_fn = batch_aug_fn


class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats