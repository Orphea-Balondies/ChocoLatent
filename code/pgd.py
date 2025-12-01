import torch
import lpips as LPIPS
import torchvision.transforms as T
from torch import optim
from utils import PGDStepResult
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
to_pil = T.ToPILImage()

def pgd(X, model, iters=200, max_img_Dis=15, max_img_lpips=15, initial_lr=0.001, eps=0.1, **kwargs):
    print(f'X.shape:{X.shape},X.dtype:{X.dtype},maxDis{max_img_Dis},iters{iters}')
    X_adv = X.clone().detach() + (torch.rand(*X.shape).to(X)*2*eps-eps)
    X_adv.requires_grad_(True)
    latent_X = model.encode(X_adv).latent_dist.mean.detach()

    #noise_image = (torch.rand_like(X) * eps - eps / 2) + X.mean(dim=(2, 3), keepdim=True)
    #noise_image = (torch.rand_like(X) *2* eps - eps) +torch.where(X.mean(dim=(2, 3), keepdim=True)>0, X.mean(dim=(2, 3), keepdim=True)+1-eps, X.mean(dim=(2, 3), keepdim=True)-1+eps)

    pbar = tqdm(range(iters+1))

    loss_fn_alex = LPIPS.LPIPS(net='alex').cuda()

    optimizer = optim.SGD(
            [X_adv],
            lr=initial_lr,           # 学习率可以比Adam大一些
            momentum=0.9,       # 动量，帮助加速收敛
            weight_decay=0      # 不需要权重衰减
        )

    for i in pbar:
        adv_img_lpips = loss_fn_alex(X, X_adv)
        adv_img_Dis = (X - X_adv).norm()#torch.mean((X - X_adv) ** 2)
        latent_X_adv = model.encode(X_adv).latent_dist.mean
        feat_Dis = (latent_X-latent_X_adv).norm()# torch.mean((latent_X-latent_X_adv)** 2)
        #feat_cos = F.cosine_similarity(latent_X.flatten(),latent_X_adv.flatten(), dim=-1).mean()
        feat_cos = F.cosine_similarity(latent_X,latent_X_adv, dim=1).mean()
        decoded_X_adv = model.decode(latent_X_adv).sample
        decoded_img_lpips = loss_fn_alex(X, decoded_X_adv)
        decoded_img_Dis = (decoded_X_adv-X).norm()#torch.mean((decoded_X_adv-X)** 2)

        if 'step_collector' in kwargs:
            metrics = {
                'adv_img_lpips': adv_img_lpips.cpu().detach().item(),
                'adv_img_Dis': adv_img_Dis.cpu().detach().item(),
                'feat_Dis': feat_Dis.cpu().detach().item(),
                'feat_cos': feat_cos.cpu().detach().item(),
                #'decoded_X_adv': (decoded_X_adv/2+0.5).cpu().detach().numpy(),
                'decoded_img_lpips': decoded_img_lpips.cpu().detach().item(),
                'decoded_img_Dis': decoded_img_Dis.cpu().detach().item()
            }
            kwargs["step_collector"].record_step(i, metrics)

        optimizer.zero_grad()
        #loss = -kwargs.r_f_c*feat_cos + kwargs.r_i_d*max(adv_img_Dis-max_img_Dis,0)
        loss = kwargs["r_f_c"]*feat_cos + \
            kwargs["r_i_d"]*max(adv_img_Dis-max_img_Dis,0) + kwargs["r_i_l"]*max(adv_img_lpips-max_img_lpips,0)#-kwargs["r_f_c"]*(1-feat_cos)

        if i % (len(pbar)//4) == 0:
            pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step lr: {optimizer.param_groups[0]['lr']:.4}")
            print(f"'adv_img_lpips': {adv_img_lpips.item():.3}|\
                'adv_img_Dis': {adv_img_Dis.item():.3}|\
                'feat_Dis': {feat_Dis.item():.3}|\
                'feat_cos': {feat_cos.item():.3}|\
                'decoded_img_lpips': {decoded_img_lpips.item():.3}|\
                'decoded_img_Dis': {decoded_img_Dis.item():.3}", flush=True)
            #print(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}", flush=True)
            pbar.update(len(pbar)//4)

        if i == iters:
            x=X.squeeze().detach().cpu().numpy()
            x_adv=X_adv.squeeze().detach().cpu().numpy()
            decoded_x_adv=decoded_X_adv.squeeze().detach().cpu().numpy()
            adv_img_psnr = psnr(x, x_adv,data_range=2.0)
            adv_img_ssim = ssim(x, x_adv,data_range=2.0,channel_axis=0)
            decoded_img_psnr = psnr(x, decoded_x_adv,data_range=2.0)
            decoded_img_ssim = ssim(x, decoded_x_adv,data_range=2.,channel_axis=0)
            print(f"'adv_img_psnr': {adv_img_psnr:.3}| \
                    'adv_img_ssim': {adv_img_ssim:.3}| \
                    'decoded_img_psnr': {decoded_img_psnr:.3}| \
                    'decoded_img_ssim': {decoded_img_ssim:.3}|\
                    'adv_img_lpips': {adv_img_lpips.item():.3}|\
                    'adv_img_Dis': {adv_img_Dis.item():.3}|\
                    'feat_Dis': {feat_Dis.item():.3}|\
                    'feat_cos': {feat_cos.item():.3}|\
                    'decoded_img_lpips': {decoded_img_lpips.item():.3}|\
                    'decoded_img_Dis': {decoded_img_Dis.item():.3}")
            break

        loss.backward()


        optimizer.step()
        with torch.no_grad():
            X_adv.data = torch.clamp(X_adv, min=X - eps, max=X + eps)
            X_adv.data = torch.clamp(X_adv, min=-1, max=1)
    #X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
    return X_adv

def loss_tl(model,Dis,X,X_adv,tgt):
    latent_X_adv = model.encode(X_adv).latent_dist.mean
    latent_Dis = (latent_X_adv-tgt).norm()
    loss = 400*Dis + latent_Dis
    return loss, latent_Dis
def loss_tp(model,Dis,X,X_adv,tgt):
    latent_X_adv = model.encode(X_adv).latent_dist.mean
    decoded_X_adv = model.decode(latent_X_adv).sample
    decoded_Dis = (decoded_X_adv-tgt).norm()
    loss = 20*Dis + decoded_Dis
    return loss, decoded_Dis
def loss_n(model,Dis,X,X_adv,tgt=None):
    latent_X_adv = model.encode(X_adv).latent_dist.mean
    decoded_X_adv = model.decode(latent_X_adv).sample
    decoded_Dis = (X-decoded_X_adv).norm()
    loss = 20*Dis - decoded_Dis
    return loss, decoded_Dis
def loss_clip(model,Dis,X,X_adv,tgt):
    latent_X_adv = model.encode(X_adv).latent_dist.mean
    decoded_X_adv = model.decode(latent_X_adv).sample
    pixel_values = [X,decoded_X_adv]
    outputs = model(pixel_values=pixel_values)
    X_embeds,decoded_X_embeds = outputs.image_embeds
    cosloss = F.cosine_similarity(X_embeds,decoded_X_embeds, dim=-1)
                                                                              