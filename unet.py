import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class input_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(input_conv, self).__init__()
        self.inp_conv = double_conv(in_c, out_c)

    def forward(x):
        x = self.inp_conv(x)

        return x

class double_conv(nn.Module):

    def __init__(self, in_c, out_c):
        super(double_conv, self).__init__()
        self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size=3),
        self.conv_2 = nn.Conv2d(out_c, out_c, kernel_size=3),


    def forward(self, x):

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
    
        return x

class up(nn.Module):
    def __init__(self, in_c, out_c):
        super(up, self).__init__()
        self.up_trans = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2)
        self.up_conv = double_conv(in_c, out_c)

    def forward(self, x1, x2):

        x1 = self.up_trans(x)
        x = self.up_conv(torch.cat([x1, x2], dim=1))
        return x

class down(nn.Module):
    def __init__(self, in_c, out_c):
        super(down, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_c, out_c)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)

        return x

class last_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(last_conv,self).__init__()
        self.out = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.out(x)

        return x
# def crop_img(tensor , target_tensor):

#     target_size = target_tensor.size()[2]
#     print(f"target size = {target_size}")

#     tensor_size = tensor.size()[2]
#     print(f"tensor size = {tensor_size}")

#     delta = tensor_size - target_size
#     print(f"delta = {delta}")

#     delta =delta // 2
#     print(f"delta / 2 = {delta}")

#     return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):

    def __init__(self, channels, classes):
        super(UNet, self).__init__()

        self.inp = input_conv(channels, 64)
        self.down_conv_1 = down(64, 128)
        self.down_conv_2 = down(128, 256)
        self.down_conv_3 = down(256, 512)
        self.down_conv_4 = down(512, 1024)


        self.up_conv_1 = up(1024, 512)
        self.up_conv_2 = up(512, 256)
        self.up_conv_3 = up(256,128)
        self.up_conv_4 = up(128, 64)
        
        self.out = last_conv(64, classes) 
        


    def forward(self, image):
        #encoder 
        #layer 1
        x1 = self.inp(image)
      
        #layer 2
        x2 = self.down_conv_1(x1)
       
        #layer 3
        x3 = self.down_conv_2(x2)

        #layer 4
        x4 = self.down_conv_4(x3)

        #layer 5
        x5 = self.down_conv_5(x4)

        #decoder 
   
        x = self.up_conv_1(x5, x4)
        x = self.up_conv_2(x, x3)
        x = self.up_conv_3(x, x2)
        x = self.up_conv_4(x, x1)
        x = self.out(x)


        return x

import click
import datasets
import mask_gen

@click.command()
@click.option('--job_desc', type=str, default='pascalaug_deeplab2i_lr3e-5_cutmix_semisup_106_split0')
@click.option('--dataset', type=click.Choice(['camvid', 'cityscapes', 'pascal', 'pascal_aug', 'isic2017']),
              default='pascal_aug')
@click.option('--model', type=click.Choice(['mean_teacher', 'pi']), default='mean_teacher')
@click.option('--arch', type=str, default='resnet101_deeplab_imagenet')
@click.option('--freeze_bn', is_flag=True, default=False)
@click.option('--opt_type', type=click.Choice(['adam', 'sgd']), default='adam')
@click.option('--sgd_momentum', type=float, default=0.9)
@click.option('--sgd_nesterov', is_flag=True, default=False)
@click.option('--sgd_weight_decay', type=float, default=5e-4)
@click.option('--learning_rate', type=float, default=1e-4)
@click.option('--lr_sched', type=click.Choice(['none', 'stepped', 'cosine', 'poly']), default='none')
@click.option('--lr_step_epochs', type=str, default='')
@click.option('--lr_step_gamma', type=float, default=0.1)
@click.option('--lr_poly_power', type=float, default=0.9)
@click.option('--teacher_alpha', type=float, default=0.99)
@click.option('--bin_fill_holes', is_flag=True, default=False)
@click.option('--crop_size', type=str, default='100,100')
@click.option('--aug_hflip', is_flag=True, default=False)
@click.option('--aug_vflip', is_flag=True, default=False)
@click.option('--aug_hvflip', is_flag=True, default=False)
@click.option('--aug_scale_hung', is_flag=True, default=False)
@click.option('--aug_max_scale', type=float, default=1.0)
@click.option('--aug_scale_non_uniform', is_flag=True, default=False)
@click.option('--aug_rot_mag', type=float, default=0.0)
@click.option('--mask_mode', type=click.Choice(['zero', 'mix']), default='mix')
@click.option('--mask_prop_range', type=str, default='0.5')
@click.option('--boxmask_n_boxes', type=int, default=1)
@click.option('--boxmask_fixed_aspect_ratio', is_flag=True, default=False)
@click.option('--boxmask_by_size', is_flag=True, default=False)
@click.option('--boxmask_outside_bounds', is_flag=True, default=False)
@click.option('--boxmask_no_invert', is_flag=True, default=False)
@click.option('--cons_loss_fn', type=click.Choice(['var', 'bce', 'kld', 'logits_var', 'logits_smoothl1']), default='var')
@click.option('--cons_weight', type=float, default=1.0)
@click.option('--conf_thresh', type=float, default=0.97)
@click.option('--conf_per_pixel', is_flag=True, default=False)
@click.option('--rampup', type=int, default=-1)
@click.option('--unsup_batch_ratio', type=int, default=1)
@click.option('--num_epochs', type=int, default=5)
@click.option('--iters_per_epoch', type=int, default=-1)
@click.option('--batch_size', type=int, default=5)
@click.option('--n_sup', type=int, default=8)
@click.option('--n_unsup', type=int, default=2)
@click.option('--n_val', type=int, default=-1)
@click.option('--split_seed', type=int, default=12345)
@click.option('--split_path', type=click.Path(readable=True, exists=True))
@click.option('--val_seed', type=int, default=131)
@click.option('--save_preds', is_flag=True, default=False)
@click.option('--save_model', is_flag=True, default=False)
@click.option('--num_workers', type=int, default=4)
def experiment(job_desc, dataset, model, arch, freeze_bn,
               opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
               learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
               teacher_alpha, bin_fill_holes,
               crop_size, aug_hflip, aug_vflip, aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,
               mask_mode, mask_prop_range,
               boxmask_n_boxes, boxmask_fixed_aspect_ratio, boxmask_by_size, boxmask_outside_bounds, boxmask_no_invert,
               cons_loss_fn, cons_weight, conf_thresh, conf_per_pixel, rampup, unsup_batch_ratio,
               num_epochs, iters_per_epoch, batch_size,
               n_sup, n_unsup, n_val, split_seed, split_path, val_seed, save_preds, save_model, num_workers):
    params = locals().copy()

    if mask_mode == 'zero':
        mask_mix = False
    elif mask_mode == 'mix':
        mask_mix = True
    else:
        raise ValueError('Unknown mask_mode {}'.format(mask_mode))
    del mask_mode

    ds_dict = datasets.load_dataset(dataset, n_val, val_seed, n_sup, n_unsup, split_seed, split_path)
    
    ds_src = ds_dict['ds_src']
    ds_tgt = ds_dict['ds_tgt']
    tgt_val_ndx = ds_dict['val_ndx_tgt']
    src_val_ndx = ds_dict['val_ndx_src'] if ds_src is not ds_tgt else None
    test_ndx = ds_dict['test_ndx_tgt']
    sup_ndx = ds_dict['sup_ndx']
    unsup_ndx = ds_dict['unsup_ndx']

    n_classes = ds_src.num_classes

    
    print('Loaded data')


    if ':' in mask_prop_range:
        a, b = mask_prop_range.split(':')
        mask_prop_range = (float(a.strip()), float(b.strip()))
        del a, b
    else:
        mask_prop_range = float(mask_prop_range)

    mask_generator = mask_gen.BoxMaskGenerator(prop_range=mask_prop_range, n_boxes=boxmask_n_boxes,
                                               random_aspect_ratio=not boxmask_fixed_aspect_ratio,
                                               prop_by_area=not boxmask_by_size, within_bounds=not boxmask_outside_bounds,
                                               invert=not boxmask_no_invert)

    train_sup_ds = ds_src.dataset(labels=True, mask=False, xf=False, pair=False,
                                    pipeline_type='cv')
    train_unsup_ds = ds_src.dataset(labels=False, mask=True, xf=False, pair=False, pipeline_type='cv')


    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    BLOCK_SIZE = (1, 1)

    collate_fn = seg_data.SegCollate(BLOCK_SIZE)
    mask_collate_fn = seg_data.SegCollate(BLOCK_SIZE, batch_aug_fn=add_mask_params_to_batch)

    train_sup_loader = torch.utils.data.DataLoader(train_sup_ds, batch_size, sampler=sup_sampler,
                                                   collate_fn=collate_fn, num_workers=num_workers)
    if cons_weight > 0.0:
        unsup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(unsup_ndx))
        train_unsup_loader_0 = torch.utils.data.DataLoader(train_unsup_ds, batch_size, sampler=unsup_sampler,
                                                           collate_fn=mask_collate_fn, num_workers=num_workers)
        if mask_mix:
            train_unsup_loader_1 = torch.utils.data.DataLoader(train_unsup_ds, batch_size, sampler=unsup_sampler,
                                                               collate_fn=collate_fn, num_workers=num_workers)
        else:
            train_unsup_loader_1 = None
    else:
        train_unsup_loader_0 = None
        train_unsup_loader_1 = None


if __name__ == "__main__":
    experiment()


    image = torch.rand((3, 572, 572))
    model = UNet(3, 1)
    #print(model(image))
     

