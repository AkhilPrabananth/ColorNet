import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from fastai.data.external import untar_data, URLs

def main(args):
    path=args.data

    if(path=="coco"):
        coco_path = untar_data(URLs.COCO_SAMPLE)
        coco_path = str(coco_path) + "/train_sample"
        total=10_000
        val=8000
    else:
        total=1026
        val=800
        
        
    paths = glob.glob(path + "/*.png") 
    np.random.seed(123)
    paths_subset = np.random.choice(paths, total, replace=False) 
    rand_idxs = np.random.permutation(total)
    train_idxs = rand_idxs[:val]
    val_idxs = rand_idxs[val:] 
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    
    SIZE = 256
    
    class ColorizationDataset(Dataset):
        def __init__(self, paths, split='train'):
            if split == 'train':
                self.transforms = transforms.Compose([
                    transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                    transforms.RandomHorizontalFlip(), # A little data augmentation!
                ])
            elif split == 'val':
                self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

            self.split = split
            self.size = SIZE
            self.paths = paths

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transforms(img)
            img = np.array(img)
            img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
            img_lab = transforms.ToTensor()(img_lab)
            L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
            ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

            return {'L': L, 'ab': ab}

        def __len__(self):
            return len(self.paths)

    def make_dataloaders(batch_size=args.batch_size, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
        dataset = ColorizationDataset(**kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                                pin_memory=pin_memory)
        return dataloader
    
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
        
    class MainModel(nn.Module):
        def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                    beta1=0.5, beta2=0.999, lambda_L1=100.,unet=True):
            super().__init__()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lambda_L1 = lambda_L1

     
            self.net_G = net_G.to(self.device)
            self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
            self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
            self.L1criterion = nn.L1Loss()
            self.unet=unet
            self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
            self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

        def set_requires_grad(self, model, requires_grad=True):
            for p in model.parameters():
                p.requires_grad = requires_grad

        def setup_input(self, data):
            self.L = data['L'].to(self.device)
            self.ab = data['ab'].to(self.device)

        def forward(self):
            if(not self.unet):
                self.fake_color = self.net_G(self.L)['pred_ab']
            else:
                self.fake_color = self.net_G(self.L)
                
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
        
            
        def forward(self, L, prev_ab = None, stm=None, ltm=None):
            if(prev_ab is None):
                n,c,h,w = L.shape
                prev_ab = torch.zeros(n, 2, h, w, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
            x = torch.concat([L, prev_ab], dim=1)
            part1 = self.unet_part1(x)
            pred_ab = self.unet_part2(part1)
            if (stm is None):
                stm, ltm = self.lstm(pred_ab)
            else:
                stm, ltm = self.lstm(pred_ab,stm,ltm)
            return {'pred_ab': pred_ab, 'stm': stm, 'ltm': ltm}
             
            
    class AverageMeter:
        def __init__(self):
            self.reset()

        def reset(self):
            self.count, self.avg, self.sum = [0.] * 3

        def update(self, val, count=1):
            self.count += count
            self.sum += count * val
            self.avg = self.sum / self.count

    def create_loss_meters():
        loss_D_fake = AverageMeter()
        loss_D_real = AverageMeter()
        loss_D = AverageMeter()
        loss_G_GAN = AverageMeter()
        loss_G_L1 = AverageMeter()
        loss_G = AverageMeter()

        return {'loss_D_fake': loss_D_fake,
                'loss_D_real': loss_D_real,
                'loss_D': loss_D,
                'loss_G_GAN': loss_G_GAN,
                'loss_G_L1': loss_G_L1,
                'loss_G': loss_G}

    def update_losses(model, loss_meter_dict, count):
        for loss_name, loss_meter in loss_meter_dict.items():
            loss = getattr(model, loss_name)
            loss_meter.update(loss.item(), count=count)

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

    def visualize(model, data):
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(data)
            model.forward()
        model.net_G.train()
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = lab_to_rgb(L, real_color)
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(L[i][0].cpu(), cmap='gray')
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(real_imgs[i])
            ax.axis("off")
        plt.show()
        fig.savefig(f"colorization_{time.time()}.png")

    def log_results(loss_meter_dict):
        for loss_name, loss_meter in loss_meter_dict.items():
            print(f"{loss_name}: {loss_meter.avg:.5f}")   
            
            
    def train_model(model, train_dl, epochs, display_every=200):
        data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
        for e in range(epochs):
            loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
            i = 0                                  # log the losses of the complete network
            for data in train_dl:
                model.setup_input(data)
                model.optimize()
                update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
                i += 1
                if i % display_every == 0:
                    print(f"\nEpoch {e+1}/{epochs}")
                    print(f"Iteration {i}/{len(train_dl)}")
                    log_results(loss_meter_dict) # function to print out the losses
                    visualize(model, data, save=False) # function displaying the model's outputs
        
    def build_res_unet(n_input=1, n_output=2, size=256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
        net_G = DynamicUnet(body, n_output, (size, size)).to(device)
        return net_G
    
    def pretrain_generator(net_G, train_dl, opt, criterion, epochs, unet=True):
        for e in range(epochs):
            loss_meter = AverageMeter()
            for data in train_dl:
                L, ab = data['L'].to(device), data['ab'].to(device)
                if not unet:
                    preds = net_G(L)['pred_ab']
                else:
                    preds = net_G(L)
                loss = criterion(preds, ab)
                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_meter.update(loss.item(), L.size(0))

            print(f"Epoch {e + 1}/{epochs}")
            print(f"L1 Loss: {loss_meter.avg:.5f}")
            
            
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    
    print("\nPretraining the U-Net part for Coloring\n")
    
    
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()        
    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    model = MainModel(net_G=net_G)
    train_model(model, train_dl, 20)
    torch.save(model.net_G.state_dict(), './weights/unet.pt')
    
    print("\nTraining ColorNet\n")

    net_G = ColorNet('./weights/unet.pt',True,True)
    model = MainModel(net_G=net_G,unet=False)
    train_model(model, train_dl, 40)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='ColorNet Pretraining')
    parser.add_argument('data',type=str,default='coco',help='Path to the data folder containing images')
    parser.add_argument('batch_size',type=int,default=16,help='Batch size for training')
    
    args = parser.parse_args()
    main(args)

    