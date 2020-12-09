import os 
import numpy as np
import seg_data


pascal_path = '/home/trojrobert/Documents/Thesis/CCT_MixMatch/data/VOCdevkit/VOC2012'


class PascalVOCAccessor (seg_data.SegAccessor):
    def __len__(self):
        return len(self.ds.sample_names)


def _load_names(path):

    """Read the file containing the list of the image names"""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line != '']

class PascalVOCDataSource (seg_data.DataSource):
    def __init__(self, n_val, val_rng, trainval_perm, fg_class_subset=None, augmented=False):
        #pascal_path = _get_pascal_path(exists=True)
        self.class_map = None

        if augmented:
            train_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'train_aug.txt')
            val_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'val.txt')

            train_names = _load_names(train_aug_names_path)
            val_names = _load_names(val_aug_names_path)

            #concatenate train and validation image names
            self.sample_names = list(set(train_names + val_names))
            self.sample_names.sort()

            #create index for the image names
            name_to_index = {name: name_i for name_i, name in enumerate(self.sample_names)}

            #get training and validation indexes
            self.train_ndx = np.array([name_to_index[name] for name in train_names])
            self.val_ndx = np.array([name_to_index[name] for name in val_names])

            #get file path to all images both training and validation
            self.semantic_y_paths = [os.path.join(pascal_path, 'SegmentationClassAug', '{}.png'.format(name)) for name in self.sample_names]

        else:
            print("The data is not augmented")
        
        self.test_ndx = None

        self.x_paths = [os.path.join(pascal_path, 'JPEGImages', '{}.jpg'.format(name)) for name in self.sample_names]
        self.num_classes = 21


    def dataset(self, labels, mask, xf, pair, transforms=None, pipeline_type='cv', include_indices=False):
        return PascalVOCAccessor(self, labels, mask, xf, pair, transforms=transforms, pipeline_type=pipeline_type,
                                 include_indices=include_indices)