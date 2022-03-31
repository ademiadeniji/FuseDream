import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image
from data import set_up_data_s2s
import wandb

parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
args = parser.parse_args()

INIT_ITERS = 1000
OPT_ITERS = 10000

utils.seed_rng(args.seed) 

sentence = args.text
device = args.device
use_wandb = args.wandb
dataset = args.dataset_name
loss_type = args.loss_type
experiment = args.experiment

if use_wandb:
    wandb.init(name=experiment, 
        project='fusedream',
        entity='adecyber')

if "reconstruct" in loss_type or loss_type == "joint":
    if dataset == "s2s":
        train_dataloader, val_dataloader, viz_dataloader = set_up_data_s2s()
        x = next(iter(train_dataloader))
        target = (x[0]*2).to(device)
    elif dataset == "imagenet32":
        train_data, valid_data, preprocess_func = set_up_data(dataset, "./", False)
        import pdb; pdb.set_trace()
    else:
        raise ValueError("specify dataset to sample from")

print('Generating:', sentence)
G, config = get_G(512, device) # Choose from 256 and 512
generator = FuseDreamBaseGenerator(G, config, 10, args.device, use_wandb=use_wandb, target=target, loss_type=loss_type) 
z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=INIT_ITERS, num_basis=5)

z_cllt_save = torch.cat(z_cllt).cpu().numpy()
y_cllt_save = torch.cat(y_cllt).cpu().numpy()
img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=True, augment=True, opt_iters=OPT_ITERS, optimize_y=True)
score = generator.measureAugCLIP(z, y, sentence, augment=True, num_samples=20)
print('AugCLIP score:', score)
import os
if not os.path.exists(f'./samples'):
    os.mkdir(f'./samples')
save_image(img, f'samples/{experiment}_generated_{OPT_ITERS}iters_%s_seed_%d_score_%.4f.png'%(sentence, args.seed, score))
save_image(target, f'samples/{experiment}_target_{OPT_ITERS}iters_%s_seed_%d_score_%.4f.png'%(sentence, args.seed, score))

