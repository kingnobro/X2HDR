import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.nn import init
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from ImageDataset import ImageDataset
import argparse
from torch.utils.data import DataLoader
from metrics import PSNR_weight,SSIM_weight,MAE_weight,dists_weight,lipis_weight
import pytorch_ssim
import lpips
import math


def calculate_optimized_x(img, optimized_k):
    """
    Calculate the x value based on the optimized k value and the image dynamic range.
    x represents the relative position of k within [l_min, l_max] (0.0 - 1.0).
    """
    img_temp = img.clone()

    mask_positive = img_temp > 0
    if mask_positive.any():
        min_v = torch.min(img_temp[mask_positive])
        img_temp[~mask_positive] = min_v
    else:
        min_v = 1e-5 
        img_temp[:] = min_v

    if img_temp.shape[1] == 3:
        L = img_temp[:, 0, :, :] * 0.212656 + img_temp[:, 1, :, :] * 0.715158 + img_temp[:, 2, :, :] * 0.072186
    else:
        L = img_temp
        
    img_l = torch.log2(L)
    l_min = img_l.min().item()
    l_max = img_l.max().item()
    
    if l_max - l_min < 1e-6:
        return 0.5
        
    x = (optimized_k - l_min) / (l_max - l_min)
    return x

class Exposure(nn.Module):
    def __init__(self,batch_size):
        super(Exposure, self).__init__()
        self.b = 1 / 128
        ek = torch.randn((batch_size,1,1,1), requires_grad=True)
        self.ek = torch.nn.Parameter(ek)
        self.register_parameter("ek", self.ek)

    def forward(self, img):
        ekk = 2 ** self.ek
        img1 = (img/(ekk+0.00000001) - self.b) / (1 - self.b)
        imgClamp = torch.clamp(img1,0.000000000001,1)
        imgP = (imgClamp) ** (1 / 2.2)
        return imgP

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_data = ImageDataset(csv_file=os.path.join('samples_sihdr.txt'),
                                       img_dir=r'/path/to/x2hdr',
                                       ref_dir=r'/path/to/gt_hdr',
                                       test=False)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=4)
        self.metric = config.metric
        self.device = torch.device("cuda")
        if self.metric=="MAE":
            self.loss_fn = torch.nn.L1Loss()
            self.assess = MAE_weight()
        elif self.metric=="SSIM":
            self.loss_fn = pytorch_ssim.SSIM1(window_size=11)
            self.assess = SSIM_weight()
        elif self.metric == "PSNR":
            self.loss_fn = torch.nn.MSELoss()
            self.assess = PSNR_weight()
        elif self.metric == "DISTS":
            self.loss_fn = torch.nn.L1Loss()
            self.assess = dists_weight()
        elif self.metric == "LPIPS":
            self.loss_fn = torch.nn.L1Loss()
            self.assess = lipis_weight()

        self.loss_fn.to(self.device)
        self.assess.to(self.device)

        self.save_ldr = config.save_ldr
        self.start_epoch = 0
        self.start_step = 0
        self.max_epochs = config.max_epochs
        self.model_name = "weight"
        self.output_dir = os.path.join('assessment', 'image', 'out')
        os.makedirs(self.output_dir, exist_ok=True)

    def fit(self):
        all_scores = []
        for step, sample_batched in enumerate(self.train_loader, 0):
            img, imgref,weight1,self.img_name = sample_batched['img'], sample_batched['imgref'], sample_batched['weight'], sample_batched['img_name']
            imgrefSq = imgref.squeeze()
            length = len(imgrefSq)
            weight2 = torch.cat(weight1, dim=0)
            weight_all = torch.sum(weight2, dim=0)
            weight=weight2/weight_all
            self.model = Exposure(length)
            self.model.to(self.device)

            self.initial_lr = config.lr
            self.decay_ratio = config.decay_ratio
            self.decay_interval = config.decay_interval
            self.max_epochs = config.max_epochs

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=5e-4)
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                 last_epoch=self.start_epoch - 1,
                                                 step_size=self.decay_interval,
                                                 gamma=self.decay_ratio)

            assessmentV = self.train_single_image(imgrefSq,img,weight.squeeze(),length)
            print(f"Image: {self.img_name[0]}, Score: {assessmentV}")
            all_scores.append(assessmentV)

            with torch.no_grad():
                optimized_k = self.model.ek.mean().item()
                x_value = calculate_optimized_x(img, optimized_k)
                print(f"Image: {self.img_name[0]}, Optimized k: {optimized_k:.4f}, Calculated x: {x_value:.4f}")

        valid_scores = [score for score in all_scores if not math.isnan(score)]

        if len(valid_scores) > 0:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"Average {self.metric} Score over {len(valid_scores)} samples: {avg_score}")
        else:
            print("No samples processed.")

    def train_single_image(self,img_Ref,img,weight,length):
        # initialize logging system
        std_loss_corrected = 0.0
        self.min_loss = 100000
        running_loss = 0
        # start training
        # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        #self.scheduler.step()
        for epoch in range(self.start_epoch, self.max_epochs):
            # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
            img1 = img.to(self.device)
            img_Ref1 = img_Ref.to(self.device).float()
            weight1 = weight.to(self.device)
            img_gen = self.model(img1).float()
            self.optimizer.zero_grad()
            self.loss = self.loss_fn(img_Ref1,img_gen).mean()*1000
            self.loss.backward()
            self.optimizer.step()

            delta_loss = abs(self.loss.data.item() - running_loss)
            if delta_loss < 10 ** (-200):
                break
            else:
                running_loss = self.loss.data.item()
                # format_str = ('(E:%d,) [Loss = %.4f] [Std Loss = %.4f]')
                # print(format_str % (epoch, running_loss, std_loss_corrected))
                self.scheduler.step()

        assementV = self.assess(img_gen, img_Ref1, weight1, length)

        if self.save_ldr:
            # Save the generated LDR sequences
            img_name_base = self.img_name[0]
            
            # Save reference LDRs
            ref_dir = os.path.join("ldr_sequences_output", f"{img_name_base}_ref")
            os.makedirs(ref_dir, exist_ok=True)
            for i, ldr_img in enumerate(img_Ref1):
                save_image(ldr_img, os.path.join(ref_dir, f"ref_{i:03d}.png"))
                
            # Save generated LDRs for the test image
            gen_dir = os.path.join("ldr_sequences_output", f"{img_name_base}_gen")
            os.makedirs(gen_dir, exist_ok=True)
            for i, ldr_img in enumerate(img_gen):
                save_image(ldr_img, os.path.join(gen_dir, f"gen_{i:03d}.png"))
            
            print(f"Saved LDR sequences for {img_name_base} in ldr_sequences_output")

        return assementV.data.item()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric", type=str, default="PSNR") # choose from MAE,PSNR,SSIM,LPIPS,DISTS
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_ldr", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the generated LDR sequences") 
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--decay_interval", type=int, default=2000)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=19901116)

    return parser.parse_args()

def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)




