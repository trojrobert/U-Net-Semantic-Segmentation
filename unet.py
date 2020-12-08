import torch 
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )

    return conv
    

def crop_img(tensor , target_tensor):

    target_size = target_tensor.size()[2]
    print(f"target size = {target_size}")

    tensor_size = tensor.size()[2]
    print(f"tensor size = {tensor_size}")

    delta = tensor_size - target_size
    print(f"delta = {delta}")

    delta =delta // 2
    print(f"delta / 2 = {delta}")

    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)


        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(1024, 512)


        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)

        self.up_conv_2 = double_conv(512, 256)



        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)

        self.up_conv_3 = double_conv(256, 128)


        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)

        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1

        )


    def forward(self, image):
        #encoder 
        #layer 1
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        #layer 2
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        #layer 3
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        #layer 4
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        #layer 5
        x9 = self.down_conv_5(x8)

        #decoder 
        x = self.up_trans_1(x9)
        print(f"X9 size = {x.size()}")

        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        print(f"x7 size = {x7.size()}")
        print(f"y size = {y.size()}")

        print(f"X new size = {x.size()}")


        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        print(f"Final X size = {x.size()}")
        


if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image))
     

