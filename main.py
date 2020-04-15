import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from utils import *
from datasets import *
from vgg16 import *
from histogram import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# define makeup part default lambda
lambda_makeup = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="beauty", help="name of the dataset")
parser.add_argument("--makeup_part", type=str, default="eyes", help="makeup part to be trained")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_adv", type=float, default=1.0, help="adversarial loss weight")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_per", type=float, default=0.005, help="perceptual loss weight")
parser.add_argument("--lambda_eyes", type=float, default=1, help="eyes loss weight")
parser.add_argument("--lambda_lips", type=float, default=1, help="lips loss weight")
parser.add_argument("--lambda_face", type=float, default=0.1, help="face loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.makeup_part, exist_ok=True)
os.makedirs("saved_models/%s" % opt.makeup_part, exist_ok=True)

# Losses
criterion_adv = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_perceptual = torch.nn.MSELoss()
criterion_l2 = torch.nn.MSELoss()
vgg = VGG16(requires_grad=False)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G = Generator(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    print("cuda enabled")
    G = G.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    vgg = vgg.cuda()
    criterion_adv.cuda()
    criterion_cycle.cuda()
    criterion_perceptual.cuda()
    criterion_l2.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/%s/G_%d.pth" % (opt.makeup_part, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.makeup_part, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.makeup_part, opt.epoch)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(256),
    transforms.ToTensor(),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("../beauty/", transforms_=transforms_, unaligned=True, face_part=opt.makeup_part),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../beauty/", transforms_=transforms_, unaligned=True, face_part=opt.makeup_part),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    fake_B, fake_A = G(real_A, real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.makeup_part, batches_done), normalize=False)

if opt.makeup_part == "face":
    lambda_makeup = opt.lambda_face

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_A_mask = Variable(batch["A_mask"].type(Tensor))
        real_B_mask = Variable(batch["B_mask"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G.train()

        optimizer_G.zero_grad()

        # Adversarial loss
        fake_B, fake_A = G(real_A, real_B)
        loss_adv_A = criterion_adv(D_A(fake_A), valid)
        loss_adv_B = criterion_adv(D_B(fake_B), valid)
        loss_adv = loss_adv_A + loss_adv_B

        # Perceptual loss
        loss_perceptual_A = criterion_perceptual(vgg(fake_B).relu4_1, vgg(real_A).relu4_1)
        loss_perceptual_B = criterion_perceptual(vgg(fake_A).relu4_1, vgg(real_B).relu4_1)
        loss_perceptual = loss_perceptual_A + loss_perceptual_B

        # Cycle loss
        recov_B, recov_A = G(fake_A, fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # Histogram loss
        channels_A = list(torch.split(fake_B, 1, 1))
        channels_B = list(torch.split(real_B, 1, 1))

        src_mask = real_A_mask > 0
        ref_mask = real_B_mask > 0

        src_masked_1 = torch.masked_select(channels_A[0], src_mask)
        src_masked_2 = torch.masked_select(channels_A[1], src_mask)
        src_masked_3 = torch.masked_select(channels_A[2], src_mask)
        temp_src = torch.cat([src_masked_1.unsqueeze(0), src_masked_2.unsqueeze(0)], 0)
        src_masked = torch.cat([temp_src, src_masked_3.unsqueeze(0)], 0)

        ref_masked_1 = torch.masked_select(channels_B[0], ref_mask)
        ref_masked_2 = torch.masked_select(channels_B[1], ref_mask)
        ref_masked_3 = torch.masked_select(channels_B[2], ref_mask)
        temp_ref = torch.cat([ref_masked_1.unsqueeze(0), ref_masked_2.unsqueeze(0)], 0)
        ref_masked = torch.cat([temp_ref, ref_masked_3.unsqueeze(0)], 0)

        src_matched = histogram_matching_cuda(src_masked.unsqueeze(0), ref_masked.unsqueeze(0))
        loss_makeup = criterion_l2(src_masked.unsqueeze(0), src_matched)

        # Total loss
        loss_G = opt.lambda_adv * loss_adv + opt.lambda_cyc * loss_cycle + opt.lambda_per * loss_perceptual + lambda_makeup * loss_makeup

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_adv(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_adv(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = loss_real + loss_fake

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_adv(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_adv(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = loss_real + loss_fake

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = loss_D_A + loss_D_B

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, makeup: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_adv.item(),
                loss_cycle.item(),
                loss_perceptual.item(),
                loss_makeup.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    # lr_scheduler_G.step()
    # lr_scheduler_D_A.step()
    # lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/%s/G_%d.pth" % (opt.makeup_part, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.makeup_part, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.makeup_part, epoch))