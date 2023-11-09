import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from torchvision import utils as vutils
import dataset_process
from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_device_batch_size', type=int, default=128)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.65)
    parser.add_argument('--total_epoch', type=int, default=1200)
    parser.add_argument('--warmup_epoch', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--vit_type', type=str, default="vit-tiny")
    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    if args.data_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                      transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                                    transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    elif args.data_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                      transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                    transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    elif args.data_name == 'stl10':
        train_dataset = torchvision.datasets.STL10('data', transform=Compose([ToTensor(), Normalize(0.5, 0.5)]), split='train', download=True)
        val_dataset = torchvision.datasets.STL10('data', split='test', download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    #dataloader,num_class,val_dataset  = load_data(args.data_dir, args.data_name,True,  args.image_size, load_batch_size, 2)
    writer = SummaryWriter(args.save_path + os.path.join('logs',args.data_name, 'mixup_feature_pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.vit_type == "vit_tiny":
        model = MAE_ViT(mask_ratio=args.mask_ratio,
                        image_size = args.image_size,
                        patch_size=args.patch_size).to(device)
    elif args.vit_type == "vit_base":
        model = MAE_ViT(mask_ratio=args.mask_ratio,
                        image_size=args.image_size,
                        patch_size=args.patch_size,
                        emb_dim = 768,
                        encoder_layer = 12,
                        encoder_head = 12,
                        decoder_layer = 6,
                        decoder_head = 12,).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()

    # 检查是否存在之前的检查点文件
    if os.path.exists(args.save_path + 'checkpoint.pth'):
        # 加载之前的检查点
        checkpoint = torch.load(args.save_path + 'checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("load previous model epoch {}".format(checkpoint['epoch']))
    else:
        start_epoch = 0

    for e in range(start_epoch, args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            mixup_img = mixup_sobel_hog(img,lamda =args.lamda)
            predicted_img, mask = model(img)
            #loss = torch.mean((predicted_img - mixup_img) ** 2 * mask) / args.mask_ratio
            loss = torch.mean((predicted_img - mixup_img) ** 2)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(48)])
            val_img = val_img.to(device)
            sobel_val_img = mixup_sobel_hog(val_img,lamda =args.lamda)
            predicted_val_img, mask = model(val_img)
            grid_4 = vutils.make_grid(predicted_val_img, nrow=8, padding=2, normalize=True)
            predicted_val_img = predicted_val_img * mask + sobel_val_img * (1 - mask)
            # img = torch.cat([sobel_val_img * (1 - mask), predicted_val_img, sobel_val_img], dim=0)
            # img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            # img_1 = torch.cat([val_img * (1 - mask),val_img,val_img * (1 - mask)], dim=0)
            # img_1 = rearrange(img_1, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            # writer.add_image('mae_image', (img + 1) / 2, global_step=e)
            # vutils.save_image(img.to(torch.device('cpu')), args.save_path + "gen_image.png")
            # vutils.save_image(img_1.to(torch.device('cpu')), args.save_path + "masked_image.png")
            # 创建一个图像网格
            grid = vutils.make_grid(val_img, nrow=8, padding=2, normalize=True)
            grid_1 = vutils.make_grid(sobel_val_img, nrow=8, padding=2, normalize=True)
            grid_2 = vutils.make_grid(predicted_val_img, nrow=8, padding=2, normalize=True)
            grid_3 = vutils.make_grid(val_img * (1 - mask), nrow=8, padding=2, normalize=True)

            writer.add_image('mask_image', grid_3, global_step=e)
            writer.add_image('pred_image', grid_2, global_step=e)
            writer.add_image('targ_image', grid_1, global_step=e)
            writer.add_image('orig_image', grid, global_step=e)
            vutils.save_image(grid.to(torch.device('cpu')), args.save_path + "real_image.png")
            vutils.save_image(grid_1.to(torch.device('cpu')), args.save_path + "target_image.png")
            vutils.save_image(grid_2.to(torch.device('cpu')), args.save_path + "pred_image.png")
            vutils.save_image(grid_3.to(torch.device('cpu')), args.save_path + "masked_image.png")
            vutils.save_image(grid_4.to(torch.device('cpu')), args.save_path + "pred_image_1.png")
        ''' save model '''

        torch.save(model, args.save_path + args.model_path)
        # save checkpoint
        # 保存检查点
        checkpoint = {
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, args.save_path + 'checkpoint.pth')