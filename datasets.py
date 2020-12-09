
import pickle
import numpy as np
import pascal_voc_dataset
import torch.utils.data

def load_dataset(dataset, n_val, val_seed, n_sup, n_unsup, split_seed, split_path):
    val_rng = np.random.RandomState(val_seed)


    #check for trainval_pem
    if split_path is not None:
        trainval_perm = pickle.load(open(split_path, 'rb'))
    else:
        trainval_perm = None

    if dataset == 'pascal_aug':
        ds_src = pascal_voc_dataset.PascalVOCDataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm, augmented=True)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx

    else:
        print("Wrong dataset")

    # Get training and validation sample indices
    split_rng = np.random.RandomState(split_seed)
        
    if split_path is not None:
        # The supplied split will have been used to shuffle the training samples, so
        # set train_perm to be the identity
        train_perm = np.arange(len(ds_src.train_ndx))
    else:
        # Random order
        train_perm = split_rng.permutation(len(ds_src.train_ndx))



    if ds_tgt is ds_src:
        if n_sup != -1:
            sup_ndx = ds_src.train_ndx[train_perm[:n_sup]]
            if n_unsup != -1:
                unsup_ndx = ds_src.train_ndx[train_perm[n_sup:n_sup + n_unsup]]
            else:
                unsup_ndx = ds_src.train_ndx[train_perm]
        else:
            sup_ndx = ds_src.train_ndx
            if n_unsup != -1:
                unsup_ndx = ds_src.train_ndx[train_perm[:n_unsup]]
            else:
                unsup_ndx = ds_src.train_ndx
    else:
        if n_sup != -1:
            sup_ndx = ds_src.train_ndx[train_perm[:n_sup]]
        else:
            sup_ndx = ds_src.train_ndx
        if n_unsup != -1:
            unsup_perm = split_rng.permutation(len(ds_tgt.train_ndx))
            unsup_ndx = ds_tgt.train_ndx[unsup_perm[:n_unsup]]
        else:
            unsup_ndx = ds_tgt.train_ndx

    return dict(
        ds_src=ds_src, ds_tgt=ds_tgt,
        val_ndx_tgt=val_ndx_tgt, val_ndx_src=val_ndx_src, test_ndx_tgt=test_ndx_tgt,
        sup_ndx=sup_ndx, unsup_ndx=unsup_ndx,
    )