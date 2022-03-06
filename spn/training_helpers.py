import random

import torch
import torch.nn.functional as f
from torch import nn

from spn.dataset_loader import UtteranceBatch


class MaskedLoss(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        # "flatten" all logits and targets by putting all subsequences together
        # pred = torch.log1p(pred).contiguous().view(-1)
        # target = torch.log1p(target).contiguous().view(-1)
        pred = torch.log1p(pred).contiguous().view(-1)
        target = torch.log1p(target).contiguous().view(-1)
        # pred = torch.log10(1 + pred).contiguous().view(-1)
        # target = torch.log10(1 + target).contiguous().view(-1)
        mask = mask.view(-1)
        pred = (mask * pred.T).T
        return self.mse(pred, target)


class LabelSmoothingLossAudioOld(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, mask):
        # print(pred.shape, target.shape, mask.shape)
        pred = pred.log_softmax(dim=self.dim)
        # pred: torch.Size([32, 512, 61]) target: torch.Size([32, 512])
        # Mask:torch.Size([32, 512])
        pred = (mask * pred.T).T
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class PositionShuffleLoss(nn.Module):
    mse = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    # w = torch.FloatTensor([min(1.05 ** i, 10) for i in range(POS_DIM)]).to(device)

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask.unsqueeze(2))
        idx = random.randint(0, 5)
        # idx = 8
        if idx == 0:
            return self.mse(pred, target)
        # elif idx == 1:
        # return self.mse(pred * self.w, target * self.w)
        if idx == 2:
            return self.mse(pred, target) * (2.0 - self.cos(pred, target)).mean()
        return (1.0 - self.cos(pred, target)).mean()


class PositionMSELoss(nn.Module):
    mse = nn.MSELoss()

    # w = torch.FloatTensor([min(1.05 ** i, 10) for i in range(POS_DIM)]).to(device)

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask.unsqueeze(2))
        return self.mse(pred, target)

        # idx = random.randint(0, 1)
        # if idx == 0:
        #     return self.mse(pred, target)
        # elif idx == 1:
        #     return self.mse(pred * self.w, target * self.w)


class CosineLoss(nn.Module):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask.unsqueeze(2))
        return (1.0 - self.cos(pred, target)).mean()


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        return self.mse(pred, target)


class MaskedSoftL1(nn.Module):
    loss = nn.SmoothL1Loss()

    def __init__(self, factor=5):
        super().__init__()
        self.factor = factor

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        return self.loss(pred / self.factor, target / self.factor)


class LabelSmoothingLossAudio(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super().__init__()
        assert 0.0 <= smoothing <= 1.0

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, mask):
        # print(pred.shape, target.shape, mask.shape)
        pred = pred.log_softmax(dim=self.dim)
        # pred: torch.Size([32, 512, 61]) target: torch.Size([32, 512])
        # Mask:torch.Size([32, 512])
        pred = (mask * pred.T).T
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def position_encode_trainer(batch: UtteranceBatch, model: nn.Module, loss_function: nn.Module):
    features_audio = batch.features.padded
    masks_audio = batch.features.masks
    features = batch.in_transcription.padded
    masks = batch.in_transcription.masks
    target = batch.position.padded
    target_mask = batch.position.masks

    result = model(features, masks, features_audio, masks_audio)
    return loss_function(result, target, target_mask)


def position_gradient_trainer(batch: UtteranceBatch, model: nn.Module, loss_function: nn.Module):
    features_audio = batch.features.padded
    masks_audio = batch.features.masks
    features = batch.in_transcription.padded
    masks = batch.in_transcription.masks

    result = model(features, masks, features_audio, masks_audio)

    target = batch.border.padded
    target_mask = batch.border.masks
    return loss_function(result, target, target_mask, batch.weight.padded)


def audio_detection_trainer(batch: UtteranceBatch, model: nn.Module, loss_function: nn.Module):
    features_audio = batch.features.padded
    masks_audio = batch.features.masks
    features_transcription = batch.in_transcription.padded
    masks_transcription = batch.in_transcription.masks
    target = batch.labels.padded

    result = model(features_transcription, masks_transcription, features_audio, masks_audio)

    batch_s, time_s, _feat_s = result.shape
    # "flatten" all logits and targets by putting all subsequences together
    flattened_result = result.contiguous().view(batch_s * time_s, -1)
    flattened_targets = target.contiguous().view(-1)
    flattened_masks = masks_audio.view(-1)

    return loss_function(flattened_result, flattened_targets, flattened_masks)


def duration_trainer(batch: UtteranceBatch, model: nn.Module, loss_function: nn.Module):
    features_audio = batch.features.padded
    masks_audio = batch.features.masks
    features = batch.in_transcription.padded
    masks = batch.in_transcription.masks

    result = model(features, masks, features_audio, masks_audio)

    target = batch.out_duration.padded
    target_mask = batch.out_duration.masks
    return loss_function(result.squeeze(2), target, target_mask)


def train(
    model,
    num_epochs,
    data_iter,
    eval_iter=None,
    loss_function=CosineLoss(),
    train_function=position_encode_trainer,
    lr_decay=0.9,
    lr=0.001,
    weight_decay=1e-5,
    repeat=0,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(optimizer)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    eval_iter = eval_iter or data_iter
    eval_iter = [eval_iter] if not isinstance(eval_iter, list) else eval_iter

    for epoch in range(1, num_epochs + 1):
        for e_iter in eval_iter:
            evaluate(model, e_iter, train_function, loss_function)

        print("Starting epoch %d, learning rate is %f" % (epoch, lr_scheduler.get_lr()[0]))
        errors = []
        for batch in data_iter:
            model.zero_grad()
            model.train()
            optimizer.zero_grad()
            loss = train_function(batch, model, loss_function)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            errors.append((loss.clone().detach().cpu().numpy(), batch))

        if repeat:
            errors = list(sorted(errors, key=lambda x: x[0])[-3:])
            print("  ", *[round(s.item(), 3) for s, _ in errors])
            for _ in range(repeat):
                for _, batch in errors:
                    model.zero_grad()
                    model.train()
                    optimizer.zero_grad()
                    loss = train_function(batch, model, loss_function)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    optimizer.step()

        lr_scheduler.step()

    for e_iter in eval_iter:
        evaluate(model, e_iter, train_function, loss_function)


def evaluate(model, data_iter, train_function=position_encode_trainer, loss_function=CosineLoss()):
    model.eval()
    total_loss = 0
    size = 0
    with torch.no_grad():
        for batch in data_iter:
            total_loss += abs(train_function(batch, model, loss_function).item())
            size += 1
    print(
        f"  Evaluation[{getattr(data_iter, 'prefix', '')}] - avg_loss: {total_loss / size:.7f} count:{size} Total loss:{total_loss:.7f}",
    )


class DivMaskedMSE(nn.Module):
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    def __init__(self, cutoff, flip=False):
        super().__init__()
        self.cutoff = cutoff
        self.flip = flip
        print("CUTOFF", cutoff)

    def forward(self, pred, target, mask, weight):
        # print(weight.shape)
        diff = torch.abs(pred - target)
        if not self.flip:
            diff = diff > (self.cutoff or random.randint(0, 3))
        else:
            diff = diff < (self.cutoff or random.randint(0, 3))

        pred = pred * weight[:, :-1]
        target = target * weight[:, :-1]

        mask_diff = mask & diff
        pred = torch.mul(pred, mask_diff)
        target = torch.mul(target, mask_diff)
        mse = self.mse(pred, target)
        # return mse

        mask_diff = mask & ~diff
        pred = torch.mul(pred, mask_diff)
        target = torch.mul(target, mask_diff)
        l1 = self.l1(pred, target)

        return mse + l1


class MaskedL1(nn.Module):
    l1 = nn.L1Loss()

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.l1(pred, target)


class MaskedThing(nn.Module):
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.log1p(f.relu(pred))
        target = torch.log1p(f.relu(target))
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target)
