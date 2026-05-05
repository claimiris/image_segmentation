# metrics for image segmentation
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

# dice coefficient: measures overlap between predicted mask and ground truth
def dice_coeff(preds, targets, thr=0.5, smooth=1e-6):
    p = (preds >= thr).float().view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (p * t).sum(1)
    return ((2*inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean().item()

# intersection over union: measures overlap between predicted mask and ground truth
def iou_score(preds, targets, thr=0.5, smooth=1e-6):
    p = (preds >= thr).float().view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()

# sensitivity and specificity
def sens_spec(preds, targets, thr=0.5, smooth=1e-6):
    p = (preds >= thr).float().view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    tp = (p * t).sum(1);  fn = ((1-p)*t).sum(1)
    tn = ((1-p)*(1-t)).sum(1); fp = (p*(1-t)).sum(1)
    sens = ((tp+smooth)/(tp+fn+smooth)).mean().item()
    spec = ((tn+smooth)/(tn+fp+smooth)).mean().item()
    return sens, spec

# hausdorff distance: measures the distance between predicted mask and ground truth
def hd95_single(pred_mask, gt_mask):
    if pred_mask.sum() == 0 and gt_mask.sum() == 0: return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0: return float(max(pred_mask.shape))
    pd = distance_transform_edt(1 - pred_mask)
    gd = distance_transform_edt(1 - gt_mask)
    return float(max(np.percentile(pd[gt_mask==1], 95),
                     np.percentile(gd[pred_mask==1], 95)))

# hausdorff distance for batch
def hd95_batch(preds, targets, thr=0.5):
    p = (preds >= thr).float().cpu().numpy()
    t = targets.cpu().numpy()
    return float(np.mean([hd95_single(p[i,0], t[i,0]) for i in range(p.shape[0])]))

# metrics tracker
class MetricsTracker:
    def __init__(self, compute_hd=False):
        self.compute_hd = compute_hd; self.reset()
    def reset(self):
        self.dice, self.iou, self.sens, self.spec, self.hd = [], [], [], [], []
    def update(self, preds, targets):
        with torch.no_grad():
            p = torch.sigmoid(preds) if preds.min() < 0 else preds
            self.dice.append(dice_coeff(p, targets))
            self.iou.append(iou_score(p, targets))
            s, sp = sens_spec(p, targets)
            self.sens.append(s); self.spec.append(sp)
            if self.compute_hd:
                self.hd.append(hd95_batch(p, targets))
    def compute(self):
        r = {'Dice': np.mean(self.dice), 'IoU': np.mean(self.iou),
             'Sens': np.mean(self.sens), 'Spec': np.mean(self.spec)}
        if self.compute_hd and self.hd: r['HD95'] = np.mean(self.hd)
        return r
