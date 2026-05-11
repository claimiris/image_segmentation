# training and validation loop for medical image segmentation
import time
import torch
import numpy as np
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from src.metrics import MetricsTracker
from src.losses import HybridLoss, DeepSupervisionLoss

# warmup cosine scheduler
class WarmupCosineScheduler:
    def __init__(self, opt, warmup, total, min_lr=1e-6):
        self.opt, self.warmup, self.total, self.min_lr = opt, warmup, total, min_lr
        self.base_lrs = [g['lr'] for g in opt.param_groups]
        self._step = 0
    def step(self):
        self._step += 1
        if self._step <= self.warmup: f = self._step / self.warmup
        else: f = 0.5 * (1 + np.cos(np.pi * (self._step - self.warmup) / (self.total - self.warmup)))
        for g, blr in zip(self.opt.param_groups, self.base_lrs): g['lr'] = self.min_lr + f * (blr - self.min_lr)
    def get_lr(self): return self.opt.param_groups[0]['lr']

# train one epoch
def train_one_epoch(model, loader, opt, loss_fn, scaler, device, is_deep_sup):
    model.train()
    run_loss, nb = 0.0, 0
    tracker = MetricsTracker()
    for imgs, msks in loader:
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True).float()
        opt.zero_grad()
        with autocast('cuda'):
            out = model(imgs)
            if is_deep_sup and isinstance(out, list):
                loss = loss_fn(out, msks)
                final_logits = out[-1]
            else:
                loss, _, _ = loss_fn(out, msks) if not is_deep_sup else (loss_fn(out, msks), None, None)
                final_logits = out
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item(); nb += 1
        tracker.update(final_logits.detach(), msks)
    return run_loss/nb, tracker.compute()

# validate one epoch
def validate(model, loader, loss_fn, device, is_deep_sup):
    model.eval()
    run_loss, nb = 0.0, 0
    tracker = MetricsTracker()
    # no gradient calculation for validation so that it does not affect the training process
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(device), msks.to(device).float()
            out = model(imgs)
            if is_deep_sup and isinstance(out, list):
                loss = loss_fn(out, msks)
                final_logits = out[-1]
            else:
                loss, _, _ = loss_fn(out, msks) if not is_deep_sup else (loss_fn(out, msks), None, None)
                final_logits = out
            run_loss += loss.item(); nb += 1
            tracker.update(final_logits, msks)
    return run_loss/nb, tracker.compute()

# train model
def train_model(model, name, train_loader, val_loader, epochs, lr, device, is_deep_sup=False):
    model.to(device)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except:
            pass
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = WarmupCosineScheduler(opt, warmup=5, total=epochs)
    scaler = GradScaler()
    loss_fn = DeepSupervisionLoss() if is_deep_sup else HybridLoss()
    
    history = {'train_loss':[], 'val_loss':[], 'train_dice':[], 'val_dice':[]}
    best_dice = 0.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        tl, tm = train_one_epoch(model, train_loader, opt, loss_fn, scaler, device, is_deep_sup)
        vl, vm = validate(model, val_loader, loss_fn, device, is_deep_sup)
        sched.step()
        
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_dice'].append(tm['Dice']); history['val_dice'].append(vm['Dice'])
        
        if vm['Dice'] > best_dice:
            best_dice = vm['Dice']
            torch.save(model.state_dict(), f'{name}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f"[{name}] Ep {epoch+1}/{epochs} | Loss: {tl:.4f}/{vl:.4f} | Dice: {tm['Dice']:.4f}/{vm['Dice']:.4f} | LR: {sched.get_lr():.2e}")

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"{name} Training Complete. Best Val Dice: {best_dice:.4f} Time: {(time.time()-t0)/60:.1f}m")
    return history
