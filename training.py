import time
import logging

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import wandb

def hopfield_retrieval(image_features, mol_features, hopfield_layer):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, hopfield_layer=hopfield_layer)
    patterns_yy = hopfield(state_patterns=mol_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)
    patterns_xy = hopfield(state_patterns=mol_features, stored_patterns=image_features, hopfield_layer=hopfield_layer)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)

    return patterns_xx, patterns_yy, patterns_xy, patterns_yx

def hopfield(state_patterns, stored_patterns, hopfield_layer):
    retrieved_patterns = hopfield_layer.forward((stored_patterns.unsqueeze(0), state_patterns.unsqueeze(0), stored_patterns.unsqueeze(0))).squeeze()
    # Row vectors -> dim=1 to normalize the row vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=1, keepdim=True)
    return retrieved_patterns


def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

    return tau * (positives + negatives)

def cloob(image_features, mol_features, inv_tau, hopfield_layer):
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, mol_features, hopfield_layer)
    identity = torch.eye(p_xx.shape[0]) > 0.5
    i = identity.to(p_xx.device)

    loss_img = infoLOOB_loss(p_xx, p_xy, i, inv_tau=inv_tau)
    loss_txt = infoLOOB_loss(p_yy, p_yx, i, inv_tau=inv_tau)

    return loss_img + loss_txt

def standard_loss(image_features, mol_features, inv_tau):

    image_features = F.normalize(image_features)
    mol_features = F.normalize(mol_features)

    identity = torch.eye(image_features.shape[0]) > 0.5
    i = identity.to(image_features.device)

    loss = infoLOOB_loss(image_features, mol_features, i, inv_tau=inv_tau)
    return loss

def get_loss(model_img, model_mol, images, mols,args, hopfield_layer = None):
    
    image_features = model_img(*images)
    mol_features = model_mol(*mols)

    if args.loss == 'cloob':
        loss = cloob(
            image_features, mol_features, model_img.logit_inv_tau, hopfield_layer)
    elif args.loss == 'standard':
        loss = standard_loss(
            image_features, mol_features, model_img.logit_inv_tau)
    else: 
        raise ValueError(" 'loss' must be either 'cloob' (with hopfield) or 'standard' (without hopfield). ")
    
    return loss

def train(model_img, model_mol, optimizer_img, optimizer_mol, 
          scaler, batch, args, hopfield = None,
          n_iter = None, tb_writer=None):

    # training for a single batch

    optimizer_img.zero_grad()
    optimizer_mol.zero_grad()

    start = time.time()
    imgs, mols = batch

    data_time = time.time() - start

    # with automatic mixed precision.
    if args.precision == "amp":
        with autocast():
            
            total_loss = get_loss(model_img, model_mol, imgs, mols, args, hopfield)
            scaler.scale(total_loss).backward()

            # Note that the two optimizers do not share common gradients, 
            # so we can unscale the gradients individually.

            scaler.unscale_(optimizer_img)
            scaler.unscale_(optimizer_mol)

            torch.nn.utils.clip_grad_norm_(model_img.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_mol.parameters(), 1.0)

            scaler.step(optimizer_img)
            scaler.step(optimizer_mol)

        scaler.update()

    else:
        total_loss = get_loss(model_img, model_mol, imgs, mols, args, hopfield)
        total_loss.backward()
        optimizer_img.step()
        optimizer_mol.step()
        
    # Note: we clamp to 4.6052 = ln(100) to prevent instability of training
    model_img.logit_inv_tau.data = torch.clamp(model_img.logit_inv_tau.data, 0, 4.6052)

    batch_time = time.time() - start

    percent_complete = 100.0 * n_iter / args.iters 
    log_str = f""

    # logging
    logging.info(
        f"Train Iters: {n_iter} [{n_iter}/{args.iters * args.batch_per_epoch} ({percent_complete:.0f}%)]\t"
        f"Loss: {total_loss.cpu().item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
        f"\tLR_imgs: {optimizer_img.param_groups[0]['lr']:5f} \tLR_mol: {optimizer_mol.param_groups[0]['lr']:5f} \tinv_tau {model_img.logit_inv_tau.data.exp():.3f}{log_str}"
    )

    # save train loss / etc.
    log_data = {
        "loss": total_loss.cpu().item(),
        "data_time": data_time,
        "batch_time": batch_time,
        "inv_tau": model_img.logit_inv_tau.data.exp().item(),
        "lr_imgs": optimizer_img.param_groups[0]["lr"],
        "lr_mols": optimizer_mol.param_groups[0]["lr"]
    }

    # log to tensorboard and/or wandb
    for name, val in log_data.items():
        name = "train/" + name
        if tb_writer is not None:
            tb_writer.add_scalar(name, val, n_iter)
        if args.use_wandb:
            wandb.log({name: val, 'step': n_iter})

    

def score(query, pattern, topk):
    
    query = F.normalize(query)
    pattern = F.normalize(pattern)

    q_p = torch.einsum('id,jd->ij', query, pattern)
    p_p = torch.einsum('id,jd->ij', pattern, pattern)
    repeats = (p_p >= 0.99999)
    match = torch.topk(q_p, k = topk, dim = -1)[1].T
    correct = torch.sum(torch.max(repeats[torch.arange(query.size(0)), match], dim = 0)[0]).item()

    return correct / query.size(0)

def eval_batch(model_img, model_mol, batch, args):
    
    imgs, mols = batch
    
    with autocast(), torch.no_grad():

        image_features = model_img(*imgs)
        mol_features = model_mol(*mols)

        image_features = F.normalize(image_features)
        mol_features = F.normalize(mol_features)

        identity = torch.eye(image_features.shape[0]) > 0.5
        i = identity.to(image_features.device)

        loss = infoLOOB_loss(image_features, mol_features, i, inv_tau = model_img.logit_inv_tau)
    
        topk_acc = [score(image_features, mol_features, topk = k) for k in [1, 5, 10]]

    return (loss, topk_acc)

def evaluate(model_img, model_mol, batch, args, n_iter = None, 
             zero_shot = True, tb_writer=None, hopfield = None):
    
    model_img.train()
    model_mol.train()

    batch_metrics = eval_batch(model_img, model_mol, batch, args)
    total_loss, topk_accs = batch_metrics

    percent_complete = 100.0 * n_iter / args.iters 

    logging.info(
        f"Train Iters: {n_iter} [{n_iter}/{args.iters} ({percent_complete:.0f}%)]\t"
        f"Loss: {total_loss.cpu().item():.6f}\t"
        f"Top1 Acc: {topk_accs[0]:.6f}\tTop5 Acc: {topk_accs[1]:.6f}\tTop10 Acc: {topk_accs[2]:.6f}\t"
    )

    if zero_shot:
        log_data = {
            "zero_shot_loss": total_loss.cpu().item(),
            "zero_shot_top1": topk_accs[0],
            "zero_shot_top5": topk_accs[1],
            "zero_shot_top10": topk_accs[2]
        }
    else:
        log_data = {
            "eval_train_loss": total_loss.cpu().item(),
            "eval_train_top1": topk_accs[0],
            "eval_train_top5": topk_accs[1],
            "eval_train_top10": topk_accs[2]
        }
    # log to tensorboard and/or wandb
    for name, val in log_data.items():
        name = "train/" + name
        if tb_writer is not None:
            tb_writer.add_scalar(name, val, n_iter)
        if args.use_wandb:
            wandb.log({name: val, 'step': n_iter})

    model_img.train()
    model_mol.train()

    return batch_metrics


# logit_inv_tau thingy
# class CLIPGeneral(nn.Module):
#     def __init__(self,
#                  init_inv_tau: float = 14.3,
#                  learnable_inv_tau: bool = True,
#                  backbone_architecture: List[str] = ['ResNet', 'MLP'],
#                  **kwargs
#                  ):
#         super().__init__()

#         self.visual = get_backbone(
#             backbone_architecture[0],
#             **kwargs.get(f"{backbone_architecture[0]}-0", kwargs))
#         self.transformer = get_backbone(
#             backbone_architecture[1],
#             **kwargs.get(f"{backbone_architecture[1]}-1", kwargs))

#         # Logit scales for the inner product in the InfoNCE loss
#         self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
#         self.logit_inv_tau.requires_grad = learnable_inv_tau