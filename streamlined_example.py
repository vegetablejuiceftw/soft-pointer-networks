from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from load import load_csv, load_files, File, UtteranceBatch
from spn.dto.base import DataTransferObject
from spn.models.soft_pointer_network import SoftPointerNetwork
from spn.tools import display_diff
import numpy as np


from tqdm.auto import tqdm
import torch.nn as nn


def fix_borders(
    generated: List[File], report_error=300
):
    output = []
    for item in generated:
        borders = item.output_timestamps.copy()

        switched = False
        prev = 0
        for i, v in enumerate(borders):
            if v < prev:
                switched = True
                after = borders[i + 1] if i + 1 < len(borders) else borders.max()
                v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001

            borders[i] = v
            prev = v

        item = item.update(output_timestamps=borders)
        output.append(item)

        # TODO: enable?
        # prev = 0
        # for i, v in enumerate(borders):
        #     assert v >= prev, f"This should never happen! {i}"
        #     prev = v

        diff = item.target_timestamps - item.output_timestamps
        if np.abs(diff).max() > report_error:
            print(f"[id:{item.source}]  [{diff.min():5.0f} {diff.max():5.0f}]  {switched}")

    return output


def generate_borders(
    model, dataset, batch_size: int = 32
):
    model.eval()
    output = []
    for batch in tqdm(dataset.batch(batch_size)):
        features_audio = batch['features_spectogram']
        features_transcription = batch['features_phonemes']

        borders_predicted = model(
            features_transcription.padded.to(model.device),
            features_transcription.masks.to(model.device),
            features_audio.padded.to(model.device),
            features_audio.masks.to(model.device),
        ).cpu().detach().numpy()

        for i, item in enumerate(batch['original']):
            item: File
            length = batch['target_timestamps'].lengths[i]
            predicted_border = borders_predicted[i, :length]
            output.append(item.update(output_timestamps=predicted_border))

    return output


def report_borders(
    generated: List[File], plotting=False
):
    winlen, winstep = generated[0].config
    ms_per_step = winstep * 1000
    diffs = [item.target_timestamps[:-1] - item.output_timestamps[:-1] for item in generated]
    diff = np.concatenate(diffs) * ms_per_step

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


class MyCustomDataset(Dataset):
    def __init__(self, files: List[File]):
        self.files = sorted(files, key=lambda x: len(x.features_spectogram))

    def __getitem__(self, index):
        return self.files[index]

    def __len__(self):
        return len(self.files)

    def features_batch_process(self, batch_features: List[np.array]) -> UtteranceBatch:
        # this is used when a list of data items is transformed into a batch
        # TODO: could we, should we use pack_padded_sequence
        padded = nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
        lens = torch.tensor([len(item) for item in batch_features])
        _b, max_len, *_f = padded.shape
        return UtteranceBatch(
            padded,
            torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1),
            lens,
        )

    def collate_fn(self, batch: List[DataTransferObject]):
        batch = sorted(batch, key=lambda x: -len(x.features_spectogram))
        result = {'original': batch}
        first = batch[0]
        for k in first.__fields__.keys():
            values = [getattr(item, k) for item in batch]
            result[k] = self.features_batch_process(values) if torch.is_tensor(values[0]) else values
        return result

    def batch(self, batch_size):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)


def train(
    model,
    num_epochs,
    data_iter,
    loss_function,
    train_function,
    eval_iter=None,
    lr_decay=0.9,
    lr=0.001,
    weight_decay=1e-5,
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
        for batch in tqdm(data_iter):
            model.zero_grad()
            model.train()
            optimizer.zero_grad()
            loss = train_function(batch, model, loss_function)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            errors.append((loss.clone().detach().cpu().numpy(), batch))

        lr_scheduler.step()

    for e_iter in eval_iter:
        evaluate(model, e_iter, train_function, loss_function)


def evaluate(model, data_iter, train_function, loss_function):
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


def position_gradient_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_timestamps']

    result = model(
        features_transcription.padded.to(model.device),
        features_transcription.masks.to(model.device),
        features_audio.padded.to(model.device),
        features_audio.masks.to(model.device),
    )
    return loss_function(result, target.padded.to(model.device), target.masks.to(model.device))


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask, *_):
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = None

    base = ".data"
    file_path = f"{base}/test_data.csv"
    test_files = load_csv(file_path, sa=False)[:limit]
    test_files = load_files(base, test_files)
    test_dataset = MyCustomDataset(test_files)

    soft_pointer_model = SoftPointerNetwork(54, 26, 256, device=device, dropout=0.2)
    soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")

    generated = generate_borders(soft_pointer_model.with_gradient, test_dataset)
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)

    torch.cuda.empty_cache()
    # SHOULD BE Evaluation[] - avg_loss: 1.7691183 count:21 Total loss:37.1514837
    # lr = 0.0000970
    # lr = 0.00000970
    lr = 0.0000000970
    train(
        soft_pointer_model.with_gradient,
        int(3),
        test_dataset.batch(32),
        MaskedMSE(),
        eval_iter=[test_dataset.batch(64)],
        train_function=position_gradient_trainer,
        lr_decay=0.98, lr=lr, weight_decay=1e-05)

    generated = generate_borders(soft_pointer_model.with_gradient, test_dataset)
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)
