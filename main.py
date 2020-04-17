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

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="beauty", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_adv", type=float, default=1.0, help="adversarial loss weight")
parser.add_argument("--lambda_per", type=float, default=0.5, help="perceptual loss weight")
parser.add_argument("--lambda_id", type=float, default=10, help="identity loss weight")
parser.add_argument("--lambda_eyes", type=float, default=1, help="eyes loss weight")
parser.add_argument("--lambda_lips", type=float, default=1, help="lips loss weight")
parser.add_argument("--lambda_face", type=float, default=0.1, help="face loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Losses
criterion_adv = torch.nn.MSELoss()
criterion_identity = torch.nn.MSELoss()
criterion_perceptual = torch.nn.MSELoss()
criterion_l2 = torch.nn.MSELoss()
vgg = VGG16(requires_grad=False)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G = Generator(input_shape, opt.n_residual_blocks)
D = Discriminator(input_shape)

if cuda:
    print("cuda enabled")
    G = G.cuda()
    D = D.cuda()
    vgg = vgg.cuda()
    criterion_adv.cuda()
    criterion_identity.cuda()
    criterion_perceptual.cuda()
    criterion_l2.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/G_%d.pth" % (opt.epoch)))
    D.load_state_dict(torch.load("saved_models/D_%d.pth" % (opt.epoch)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(128),
    transforms.ToTensor(),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("../beauty/", transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../beauty/", transforms_=transforms_, unaligned=True),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G.eval()
    real_A = Variable(imgs["A"].type(Tensor), requires_grad=False)
    real_B = Variable(imgs["B"].type(Tensor), requires_grad=False)
    real_C = Variable(imgs["C"].type(Tensor), requires_grad=False)
    real_D = Variable(imgs["D"].type(Tensor), requires_grad=False)
    fake = G(real_A, real_B, real_C, real_D)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    real_C = make_grid(real_C, nrow=5, normalize=True)
    real_D = make_grid(real_D, nrow=5, normalize=True)
    fake = make_grid(fake, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, real_B, real_C, real_D, fake), 1)
    save_image(image_grid, "images/%s.png" % (batches_done), normalize=False)

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_C = Variable(batch["C"].type(Tensor))
        real_D = Variable(batch["D"].type(Tensor))

        real_A_mask_eyes = Variable(batch["A_mask_eyes"].type(Tensor), requires_grad=False)
        real_A_mask_lips = Variable(batch["A_mask_lips"].type(Tensor), requires_grad=False)
        real_A_mask_face = Variable(batch["A_mask_face"].type(Tensor), requires_grad=False)
        real_A_mask_bg = Variable(batch["A_mask_bg"].type(Tensor), requires_grad=False)

        real_B_mask = Variable(batch["B_mask"].type(Tensor), requires_grad=False)
        real_C_mask = Variable(batch["C_mask"].type(Tensor), requires_grad=False)
        real_D_mask = Variable(batch["D_mask"].type(Tensor), requires_grad=False)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G.train()

        optimizer_G.zero_grad()

        # Adversarial loss
        fake_B = G(real_A, real_B, real_C, real_D)
        loss_adv = criterion_adv(D(fake_B), valid)

        # Perceptual loss
        loss_perceptual = criterion_perceptual(vgg(fake_B).relu4_1, vgg(real_A).relu4_1)

        # Identity loss
        src_masked, ref_masked = mask_regions(fake_B, real_A, real_A_mask_bg, real_A_mask_bg)
        loss_id = criterion_identity(src_masked, ref_masked)

        # Eyes Histogram loss
        src_masked, ref_masked = mask_regions(fake_B, real_B, real_A_mask_eyes, real_B_mask)
        src_matched = histogram_matching_cuda(src_masked.unsqueeze(0), ref_masked.unsqueeze(0))
        loss_eyes = criterion_l2(src_masked.unsqueeze(0), src_matched)

        # Lips Histogram loss
        src_masked, ref_masked = mask_regions(fake_B, real_C, real_A_mask_lips, real_C_mask)
        src_matched = histogram_matching_cuda(src_masked.unsqueeze(0), ref_masked.unsqueeze(0))
        loss_lips = criterion_l2(src_masked.unsqueeze(0), src_matched)

        # Face Histogram loss
        src_masked, ref_masked = mask_regions(fake_B, real_D, real_A_mask_face, real_D_mask)
        src_matched = histogram_matching_cuda(src_masked.unsqueeze(0), ref_masked.unsqueeze(0))
        loss_face = criterion_l2(src_masked.unsqueeze(0), src_matched)

        loss_makeup = opt.lambda_eyes * loss_eyes + opt.lambda_lips * loss_lips + opt.lambda_face * loss_face

        # Total loss
        loss_G = opt.lambda_adv * loss_adv + opt.lambda_per * loss_perceptual + loss_makeup + opt.lambda_id * loss_id

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_adv(D(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_adv(D(fake_B_.detach()), fake)
        # Total loss
        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizer_D.step()

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, identity: %f, perceptual: %f, makeup: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_adv.item(),
                loss_id.item(),
                loss_perceptual.item(),
                loss_makeup.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/G_%d.pth" % (epoch))
        torch.save(D.state_dict(), "saved_models/D_%d.pth" % (epoch))