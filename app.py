import torch
import warnings
import gradio as gr
import torchvision
from torch import nn
import numpy as np
from torch import optim

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
from skimage.color import rgb2lab, lab2rgb

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet


device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

SIZE=256

def init_weights(net, init='norm', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                          for i in range(n_down)] # the 'if' statement is taking care of not using
        
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    
class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)['pred_ab']

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()



def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


class NDIM_LSTM(nn.Module):
    
    def __init__(self,size):
        super().__init__()

        self.percent_ltr_input=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.percent_ltr_stm_wt=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.b1=nn.Parameter(torch.tensor(0.),requires_grad=False)

        self.percent_potential_ltm_stm_wt=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.percent_potential_ltm_input=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.b2=nn.Parameter(torch.tensor(0.),requires_grad=False)
        
        self.potential_ltm_stm_wt=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.potential_ltm_input=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.b3=nn.Parameter(torch.tensor(0.),requires_grad=False)
        
        self.output_stm_contri_stm_wt=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.output_stm_contri_input=nn.Parameter(torch.empty(size).normal_(mean=0.0,std=1.0),requires_grad=True)
        self.b4=nn.Parameter(torch.tensor(0.),requires_grad=False)

    def lstm_unit(self,input_value,long_memory,short_memory):
        
        long_remember_percent=torch.sigmoid((input_value*self.percent_ltr_input)+
                                            (self.percent_ltr_stm_wt*short_memory)+
                                            self.b1)
        
        potential_remember_percent=torch.sigmoid((input_value*self.percent_potential_ltm_input)+
                                                 (short_memory*self.percent_potential_ltm_stm_wt)+
                                                  self.b2)

        potential_memory = torch.tanh((short_memory * self.potential_ltm_stm_wt) + 
                                  (input_value * self.potential_ltm_input) + 
                                  self.b3)
        
        updated_long_memory = ((long_memory * long_remember_percent) + 
               (potential_remember_percent * potential_memory))

        output_percent = torch.sigmoid((short_memory * self.output_stm_contri_stm_wt) + 
                                       (input_value * self.output_stm_contri_input) + 
                                       self.b4)         
        
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        
        
        updated_long_memory = torch.tanh(updated_long_memory)
        updated_short_memory = torch.tanh(updated_short_memory)
        
        return([updated_long_memory, updated_short_memory])

    def forward(self, input, long_memory=0, short_memory=0): 
        
        return self.lstm_unit(input,long_memory,short_memory)
    
    
class ColorNet(nn.Module):
    def __init__(self,pretrained_weights=None,freeze=False,custom_decoder=False):
        super().__init__()
        self.unet_part1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=9, stride=1, padding='same').to(device)
        self.unet_part2 = build_res_unet(n_input=1, n_output=2, size=SIZE)
        self.lstm = NDIM_LSTM((2,SIZE,SIZE)).to('cuda' if torch.cuda.is_available() else 'cpu')
        if(pretrained_weights is not None):
            print(self.unet_part2.load_state_dict(torch.load(pretrained_weights)))
            if(freeze):
                for param in self.parameters():
                    param.requires_grad = False
                for param in self.unet_part1.parameters():
                    param.requires_grad = True
                for param in self.lstm.parameters():
                    param.requires_grad = True
                if(custom_decoder):
                    for param in self.unet_part2.layers[3:len(self.unet_part2.layers)].parameters():
                        param.requires_grad=True
        
    
    def forward(self, L, prev_ab = None):
        if(prev_ab is None):
            n,c,h,w=L.shape
            prev_ab=torch.zeros(n,2,h,w).to(device)
        x = torch.concat([L, prev_ab], dim=1)
        part1 = self.unet_part1(x)
        pred_ab = self.unet_part2(part1)
        stm, ltm = self.lstm(pred_ab)
        return {'pred_ab': pred_ab, 'stm': stm, 'ltm': ltm}
    
    
    
net_G = ColorNet()
model = MainModel(net_G=net_G)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

        


def translate_image(image, weight_name, save):
    if(weight_name=='op_init'):
        model.load_state_dict(torch.load('./weights/full_model_manga_finetuned_00.pt'))
    elif(weight_name=='last'):
        model.load_state_dict(torch.load('./weights/full_model_manga_finetuned_last.pt'))    
    elif(weight_name=='bc'):
        model.load_state_dict(torch.load('./weights/full_model_manga_finetuned_blackclover.pt'))
    elif(weight_name=='mah'):
        model.load_state_dict(torch.load('./weights/full_model_manga_finetuned_mah.pt'))
    elif(weight_name=='latest'):
        model.load_state_dict(torch.load('./weights/latest_weights.pt'))

        
    model.eval()
    
    transforms = torchvision.transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
    img = np.array(image)
    img = transforms(torch.tensor(img).permute(2,0,1))

    img = np.array(img.permute(1,2,0))
            

    img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
    
    img_lab = torch.tensor(img_lab).permute(2,0,1)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
    
    
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input({'L': torch.unsqueeze(L,dim=0), 'ab': torch.zeros(1, 2, SIZE, SIZE)})
        model.forward()

    fake_color = model.fake_color.detach()
    color=fake_color[0].to('cpu')
    L=L.to('cpu')
    L = (L + 1.) * 50.
    color = color * 110.
    Lab = torch.cat([L, color], dim=0)
    img_rgb = lab2rgb(Lab.permute(1,2,0))
    #print(img_rgb)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    img = Image.fromarray(img_rgb)
    if save == "True":
        img.save("output.jpg")
    return img

css = """
.translated-image {
    width: 256px !important;
    height: 256px !important;
}
"""

interface = gr.Interface(
    fn=translate_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(choices=["op_init", "bc", "mah", "last","latest"], label="Select Weights", value="op_init"),
        gr.Radio(choices=["True", "False"], label="Save Output", value="False")
    ],
    outputs=gr.Image(type="pil", label="Translated Image", elem_classes="translated-image"),
    title="Correction App",
    description="Upload an image and get the translated version. Some images may be blurry, you can tick the checkbox to sharpen them. Choose between three different models for translation.",
    allow_flagging='never',
    css=css
)

interface.launch()