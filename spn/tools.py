import contextlib
from collections import defaultdict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from fastdtw import dtw as slowdtw
from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.ticker import FormatStrFormatter
# from torchtext.legacy.data import BucketIterator

from spn.constants import DURATION_SCALER, KNOWN_LABELS, ms_per_step, POS_DIM, WIN_STEP
from spn.dataset_loader import dedupe, find_borders, Utterance
from spn.models.components import Attention


@contextlib.contextmanager
def nullcontext():
    yield None


def get_aligned_result(result: np.ndarray, labels: np.ndarray):
    step_count, _feature_count = result.shape
    labelss = labels
    # result = result + np.random.random_sample(result.shape) / 5
    # labelss = labels + np.random.random_sample(labels.shape) / 5
    # distance, path = __dtw(result, labelss, dist=dist)
    _distance, path = slowdtw(result, labelss, dist=2)
    # warped_result = np.vstack([labels[index,:] for _, index in path])
    # return warped_result, None
    # if len(path) != step_count:
    #     d, cost_matrix, acc_cost_matrix, (result_to_labels, labels_to_result) = adtw(result, labelss, dist=euclidean_norm)
    #     warped_result = np.vstack([labels[index,:] for index in labels_to_result])

    # d, cost_matrix, acc_cost_matrix, (result_to_labels, labels_to_result) = adtw(result, labelss, dist='cosine')
    # warped_result = np.vstack([labels[index,:] for index in labels_to_result])
    # print(len(result_to_labels), len(labels_to_result))
    # path = list(zip(result_to_labels, labels_to_result))

    stack = [None] * step_count
    path_for_id = []
    label_ids = np.argmax(labels, axis=1)
    for index, label_id in path:
        stack[index] = labels[label_id, :]
        path_for_id.append((index, label_ids[label_id]))

    warped_result = np.vstack(stack)
    return warped_result, path_for_id


def generate_duration_transcription(transcriptions: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Durations be scaled."""
    # durations *= DURATION_SCALER
    # return transcriptions
    stack = []
    for feature, duration in zip(transcriptions, durations):
        adj_dur = duration / ms_per_step

        steps = max((adj_dur).round().item(), 1)
        for _ in range(int(steps)):
            stack.append(feature)

    return np.vstack(stack)


def find_borders_pathed(path: list, original_mapping: list):
    borders = []
    last = path[0][1]

    for idx, label_id in path:
        if last != label_id:
            # add 0.5 to the border, as it is between the two frames
            pos = idx - 0.5
            borders.append(pos * ms_per_step)
        last = label_id

    borders_pred = np.array(borders)
    borders_truth = np.array([
        end for i, (voc, end) in enumerate(original_mapping)
        if original_mapping[min(i + 1,
                                len(original_mapping) - 1)][0] != voc
    ], )

    assert \
        borders_pred.shape == borders_truth.shape,\
        f"[error] Mapping and Output have same composition {borders_pred.shape} {borders_truth.shape}"

    diff = borders_truth - borders_pred
    return None, None, diff


duration_cache = {}


def evaluate_result(model, iterator, lower=True, duration_model=None):
    dtw_errors = []
    detection_errors = []

    print("[standard]" if (duration_model is None) else "[duration model]")
    diff_ranking = []
    diffs = []
    cache_hits = 0
    for batch in iterator:
        features_audio = batch.features.padded
        batch_s, _time_s, _feat_s = features_audio.shape
        masks_audio = batch.features.masks
        features_transcription = batch.in_transcription.padded
        masks_transcription = batch.in_transcription.masks
        labels = batch.labels.padded

        full_result = model(features_transcription, masks_transcription, features_audio, masks_audio)
        full_result = f.softmax(full_result, dim=2)

        full_result_cls = torch.argmax(full_result, dim=2)
        full_result = full_result.cpu().detach().numpy()

        for i in range(batch_s):
            idx = batch.index[i]
            length = batch.labels.lengths[i].cpu().detach().numpy()
            transcription = batch.transcription[i].cpu().detach().numpy()

            result = full_result[i, :length, :]
            result_cls = full_result_cls[i, :length]
            labels_cls = labels[i, :length]

            audio = features_audio[i, :length, :]
            truth = batch.label_vec[i].cpu().detach().numpy()

            if duration_model:
                key = batch.key[i]
                if key in duration_cache:
                    cache_hits += 1
                    transcription = duration_cache[key]
                else:
                    res_batch = duration_model.forward(
                        batch.transcription[i].unsqueeze(0),
                        None,
                        audio.unsqueeze(0),
                        None,
                    )
                    prediction = (res_batch).squeeze(0).detach().cpu().numpy() * DURATION_SCALER
                    transcription = generate_duration_transcription(transcription, prediction)
                    duration_cache[key] = transcription

            # labels is a sequence of vocals in order of their creation with no
            # duration
            warped_result, path = get_aligned_result(result, transcription)
            wmax = np.argmax(warped_result, axis=1)

            error = (labels_cls.cpu().detach().numpy() != wmax[:length]).sum()
            dtw_error = (error / length) * 100

            if dtw_error > 25:
                print(f"danger: dtw_error {dtw_error:.1f}% wrong idx:{idx}")
                print(f"- warped_result: {warped_result.shape}\n- truth:{truth.shape}")
                # f, axarr = plt.subplots(5, figsize=(24, 6))
                # axarr[0].text(5, 5, 'Audio', bbox={'facecolor': 'white', 'pad': 10})
                # axarr[0].imshow(audio.detach().cpu().numpy().T, origin="lower", aspect='auto', cmap=cm.winter)
                # axarr[1].imshow(truth.T, origin="lower", aspect='auto', cmap=cm.winter)
                # axarr[1].text(5, 5, 'truth', bbox={'facecolor': 'white', 'pad': 10})
                # axarr[2].imshow(transcription.T, origin="lower", aspect='auto', cmap=cm.winter)
                # axarr[2].text(5, 5, 'transcription', bbox={'facecolor': 'white', 'pad': 10})
                # axarr[3].imshow(warped_result.T, origin="lower", aspect='auto', cmap=cm.winter)
                # axarr[3].text(5, 5, 'warped_result', bbox={'facecolor': 'white', 'pad': 10})
                # axarr[4].imshow(result.T, origin="lower", aspect='auto', cmap=cm.winter)
                # axarr[4].text(5, 5, 'result - probs', bbox={'facecolor': 'white', 'pad': 10})

            if dtw_error < 1 and lower:
                print(f"winner: dtw_error {dtw_error:.1f}% good")
            dtw_errors.append(dtw_error)

            error = (labels_cls != result_cls).sum().cpu().detach().numpy()
            detection_errors.append((error / length) * 100)

            try:
                if path:
                    _a, _b, d = find_borders_pathed(path, batch.out_map[i])
                else:
                    _a, _b, d = find_borders(wmax, batch.out_map[i])
            except Exception as e:
                print("[Exception]")
                # print(np.argmax(transcription, axis=1))
                print(
                    labels_cls.shape,
                    wmax.shape,
                    transcription.shape,
                    len(batch.out_map[i]),
                )
                raise e
            diff_ranking.append((abs(d.max()), idx))
            diffs.append(d)

    diff_ranking = sorted(diff_ranking, key=lambda x: x[0])
    print(diff_ranking[-5:])
    print(f"cache_hits: {cache_hits}")
    diff = np.concatenate(diffs)
    return dtw_errors, detection_errors, diff


def display_error(errors, name=""):
    print(f"[{name}]AVERAGE ERROR: {sum(errors) / len(errors):.2f}% COUNT:{len(errors)}")


def display_diff(errors, name="", unit="ms", plotting=False):
    errors = errors.copy()
    hist, bins = np.histogram(
        abs(errors),
        bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 9999],
    )
    hist = np.round(hist / len(errors) * 100, 2)
    hist = np.cumsum(hist)

    print(
        f"[{name}] DIFF abs mean: {abs(errors).mean():.2f}{unit} "
        f"({errors.mean():.2f}) min:{abs(errors).min():.2f}{unit} "
        f"max:{abs(errors).max():.2f}{unit}",
    )
    rows = list(zip(hist, bins, bins[1:]))
    for r in zip(rows[::2], rows[1::2]):
        s = ""
        for h, _b, e in r:
            s += f"\t{h:.2f}%\t < {e:.0f}{unit}\t"
        print(s)

    print(*[f"{h:2.2f}" for h, b, e in rows][:-2], "", sep="% ")
    # print([e for h, b, e in rows])

    if plotting:
        _f, axarr = plt.subplots(1, 2, figsize=(10, 3))
        axarr[0].bar(
            range(len(bins) - 1),
            hist,
        )
        axarr[0].set_xticklabels(bins, fontdict=None, minor=False)
        axarr[1].hist(np.clip(errors, -70, 70), bins=5)


def draw_duration(model, dataset, index):
    model.eval()
    inp = dataset[index]
    prediction = (
        model.forward(inp.in_transcription.unsqueeze(0), None, inp.features.unsqueeze(0),
                      None).squeeze(0).detach().cpu().numpy() * DURATION_SCALER
    )

    inputs = inp.features.detach().cpu().numpy()
    transcription_truth = dataset.out_vec[index].detach().cpu().numpy()
    truth = inp.out_duration.detach().cpu().numpy() * DURATION_SCALER

    total_duration = inputs.shape[0] * ms_per_step
    print(total_duration, sum(prediction))
    prediction = prediction / sum(prediction) * total_duration
    transcription_with_duration = generate_duration_transcription(
        inp.in_transcription.detach().cpu().numpy(),
        prediction,
    )

    f, axarr = plt.subplots(4, figsize=(8, 8))
    axarr[0].title.set_text("1. Audio input")
    axarr[1].title.set_text("2. Duration predictions")
    axarr[2].title.set_text("3. Phoneme occurrence predicted by scaling transcription with durations")
    axarr[3].title.set_text("4. Phoneme occurrence ground truth")

    axarr[0].imshow(inputs.T, origin="lower", aspect="auto", cmap=cm.winter)
    axarr[1].plot(prediction, "r", label="Predicted durations")
    axarr[1].plot(truth, "g", label="Actual durations")
    axarr[1].legend(loc="upper left")

    axarr[2].imshow(transcription_with_duration.T, origin="lower", aspect="auto")
    axarr[3].imshow(transcription_truth.T, origin="lower", aspect="auto")

    axarr[0].set_ylabel("Audio features")
    axarr[0].set_xlabel("Audio frame index")
    axarr[1].set_ylabel("Duration in ms")
    axarr[1].set_xlabel("Phoneme index")
    axarr[2].set_xlabel("Audio frame index")
    axarr[2].set_ylabel("Phoneme one-hot encoding")
    axarr[3].set_xlabel("Audio frame index")
    axarr[3].set_ylabel("Phoneme one-hot encoding")

    f.tight_layout()


def draw_audio(model, dataset, index):
    model.eval()
    ds = dataset[index]
    features_audio = ds.features
    transcription_truth = dataset.out_vec[index].detach().cpu().numpy()

    trans = ds.in_transcription

    res_batch = model.forward(trans.unsqueeze(0), None, features_audio.unsqueeze(0), None)

    inputs = ds.features.detach().cpu().numpy()

    prediction = (res_batch).squeeze(0).detach().cpu().numpy()
    truth = (ds.out_duration).detach().cpu().numpy() * DURATION_SCALER
    trans = trans.detach().cpu().numpy()

    _f, axarr = plt.subplots(4, figsize=(10, 4))
    axarr[0].imshow(inputs.T, origin="lower", aspect="auto", cmap=cm.winter)
    axarr[1].plot(truth, "g", label="Truth")
    axarr[1].plot(prediction, "r", label="Prediction")
    axarr[1].legend(loc="upper right")
    # axarr[2].imshow(transcription_with_duration.T, origin="lower", aspect="auto", cmap=cm.winter)
    axarr[3].imshow(transcription_truth.T, origin="lower", aspect="auto", cmap=cm.winter)


def show_audio(model, dataset, name, plot_only=False, duration_model=None):
    print(f"\n[{name}]")
    model.eval()
    # show one
    i = 36
    inp = dataset[i].features
    out_vec = dataset.out_vec[i]
    transcription = dataset.in_transcription[i]
    _res = model.forward(transcription.unsqueeze(0), None, inp.unsqueeze(0), None)
    _res = f.softmax(_res, dim=2)
    result = _res[0, :, :]
    result_maximized = result.clone().detach().cpu().numpy()
    ids = np.argmax(result_maximized, axis=1)
    result_maximized = result_maximized * 0
    for t, i in enumerate(ids):
        result_maximized[t, i] = 1
    warped_result, _path = get_aligned_result(result.detach().cpu().numpy(), transcription.detach().cpu().numpy())

    _f, axarr = plt.subplots(4, figsize=(12, 12), sharex=True)
    axarr[0].imshow(inp.cpu().numpy().T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[1].imshow(result.detach().cpu().numpy().T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[2].imshow(result_maximized.T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[3].imshow(out_vec.cpu().numpy().T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[0].title.set_text("Audio input")
    axarr[1].title.set_text("CTC phoneme probabilities")
    axarr[2].title.set_text("CTC most probable phoneme")
    axarr[3].title.set_text("Phoneme ground truth")

    _f, axarr = plt.subplots(3, figsize=(12, 8), sharex=True)
    axarr[0].imshow(result.detach().cpu().numpy().T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[1].imshow(warped_result.T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[2].imshow(out_vec.cpu().numpy().T, origin="lower", aspect="auto")  # , cmap=cm.winter)
    axarr[0].title.set_text("CTC phoneme probabilities")
    axarr[1].title.set_text("Transcription phoneme sequence alligned over CTC result with DTW")
    axarr[2].title.set_text("Phoneme ground truth")

    if plot_only:
        return

    # # # # # #
    # difference percentages
    dataset_iter = BucketIterator(
        dataset,
        batch_size=64,
        sort_key=lambda x: len(x.features),
        sort=False,
        shuffle=True,
        sort_within_batch=True,
    )
    dtw_errors, detection_errors, diff = evaluate_result(
        model,
        dataset_iter,
        lower=False,
        duration_model=duration_model,
    )
    display_error(dtw_errors, "DETECTION+DTW")
    display_error(detection_errors, "DETECTION")
    display_diff(diff, name, plotting=True)


def show_duration(model, dataset, sample_size=2000):
    model.eval()
    print("dataset len", len(dataset))

    # draw_duration(model, dataset, 0)
    # # # # # #
    # difference percentages
    diffs = []
    sums = []
    for i in sample(range(len(dataset)), min(len(dataset), sample_size)):
        inp = dataset[i].in_transcription
        inp_audio = dataset[i].features

        # durations have been scaled with DURATION_SCALER
        out = dataset[i].out_duration * DURATION_SCALER
        res = model.forward(inp.view(1, *inp.shape), None, inp_audio.unsqueeze(0), None) * DURATION_SCALER

        resc = torch.cumsum(res, dim=1).view(-1)
        res = res.detach().view(-1).cpu().numpy()

        outc = torch.cumsum(out, dim=0)
        out = out.detach().cpu().numpy()

        diff = out - res
        if diff.max() > 1000:
            print(i, diff.max())
            draw_duration(model, dataset, i)
            # continue
        diffs.append(diff)
        sums.append((outc - resc).detach().cpu().numpy())

    diff = np.concatenate(diffs)
    sums = np.concatenate(sums)
    display_diff(diff, "duration", plotting=True)
    display_diff(sums, "position")


show_duration_og = show_duration


def show_position(
    model,
    dataset,
    duration_combined_model=None,
    sample_size=2000,
    report_error=750,
    skip=False,
):
    model.eval()
    if duration_combined_model is not None:
        duration_combined_model.eval()

    # torch.cumsum(torch.ones(2 ** 14), 0).unsqueeze(1) - 1
    print("dataset len", len(dataset))

    diffs = []
    Attention(POS_DIM)

    for idx in sample(range(len(dataset)), min(len(dataset), sample_size)):
        inp: Utterance = dataset[idx]

        audio = inp.features
        transcription = inp.in_transcription
        border = inp.border

        length = audio.shape[0] + 0
        borders_predicted = model(transcription.unsqueeze(0), None, audio.unsqueeze(0), None)[0]

        prev = 0
        if duration_combined_model is None:
            new = borders_predicted.detach().cpu().numpy() * ms_per_step
            switched = False
            for i, v in enumerate(new):
                if abs(v - prev) > 500 or v < prev:
                    after = new[i + 1] if i + 1 < len(new) else prev
                    v = (prev + after) / 2
                    switched = True
                new[i] = v
                prev = v
        else:
            duration = None
            prediction_position = borders_predicted.detach().cpu().numpy() * ms_per_step
            new = prediction_position.copy()
            switched = False
            for i, v in enumerate(new):
                after = new[i + 1] if i + 1 < len(new) else v
                if v < prev or v > after:
                    if duration is None:
                        duration = ((
                            duration_combined_model(transcription.unsqueeze(0), None, audio.unsqueeze(0), None) *
                            DURATION_SCALER
                        ).view(-1).detach().cpu().numpy()[:-1])
                    v = prev + duration[i - 1]
                    switched = True
                new[i] = v
                prev = v
        b = new
        diff = border.detach().cpu().numpy() * ms_per_step - b
        if np.abs(diff).max() > report_error:
            print(f"[id:{idx:3d}]  [{diff.min():5.0f} {diff.max():5.0f}]  {length:4d} {switched}")
            if skip:
                continue
        if switched and skip:
            continue

        diffs.append(diff)

    diff = np.concatenate(diffs)
    display_diff(diff, "position", plotting=True)


def location_fix(positions, truth, durations, end_of_audio):
    difos = []
    visited = []
    for _ in range(10):
        # worst_diff, worst_index = max([abs(v - Y), i] for i, (v, Y) in enumerate(zip(positions, truth) )if i not in visited)
        worst_diff, worst_index = 0, 0
        for i, v in enumerate(positions):
            if i in visited:
                continue
            prev = positions[i - 1] if i >= 1 else 0
            after = positions[i + 1] if i + 1 < len(positions) else end_of_audio
            # y = (prev + after - 0.0001) / 2
            y = prev + durations[i]
            diff = abs(v - y)
            if diff > worst_diff:
                worst_diff, worst_index = diff, i

        if worst_diff < 300:
            continue
        visited.append(worst_index)
        difos.append([worst_diff, [positions[worst_index], truth[worst_index]], worst_index])

        i = worst_index
        v = positions[i]
        prev = positions[i - 1] if i >= 1 else 0
        after = positions[i + 1] if i + 1 < len(positions) else end_of_audio

        v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001
        positions[worst_index] = v

    prev = 0
    for i, _ in enumerate(positions):
        v = positions[i]

        if v < prev:
            after = positions[i + 1] if i + 1 < len(positions) else prev + 0.01

        if v < prev:
            v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001
        positions[i] = v
        prev = v

    prev = 0
    for i, v in enumerate(positions):
        assert v >= prev, f"This should never happen! {i}"
        prev = v
    return positions, difos


def plot(audio, transcript):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    f, axarr = plt.subplots(2, figsize=(8, 8))
    axarr[0].imshow(audio.T, origin="lower", aspect="auto", cmap=cm.winter)
    axarr[1].imshow(transcript.T, origin="lower", aspect="auto", cmap=cm.winter)
    plt.show()

def show_position_batched(
    model, dataset, duration_combined_model=None, report_error=750, plotting=False
):  # noqa: MC0001
    model.eval()
    if duration_combined_model is not None:
        duration_combined_model.eval()

    diffs = []
    label_ids = []

    for batch in dataset.batch(32):
        batch: Utterance
        features_audio = batch.features.padded
        masks_audio = batch.features.masks

        features_transcription = batch.in_transcription.padded
        masks_transcription = batch.in_transcription.masks

        # print(batch.label_file, batch.audio_file)
        # print("features_transcription", batch.in_transcription.padded.shape, batch.in_transcription.masks.sum())
        # print("features_audio", batch.features.padded.shape, batch.features.masks.sum())
        # plot(batch.features.padded.cpu().detach().numpy(), batch.in_transcription.padded.cpu().detach().numpy())
        # torch.save(batch.in_transcription.padded[0], "vanilla_transcription.pth")
        borders = batch.border.padded.cpu().detach().numpy()
        border_lengths = batch.border.lengths.cpu().detach().numpy()

        batch_s, _time_s, _feat_s = features_audio.shape

        borders_predicted = (
            model(features_transcription, masks_transcription, features_audio, masks_audio).cpu().detach().numpy()
        )

        if duration_combined_model is not None:
            duration_batch = ((
                duration_combined_model(features_transcription, masks_transcription, features_audio, masks_audio) *
                DURATION_SCALER
            ).detach().cpu().numpy())

        for i in range(batch_s):
            label_id = [l_id for l_id, ms in batch.out_map[i]]
            idx = batch.index[i]

            length = border_lengths[i]

            predicted_border = borders_predicted[i, :length]
            truth_border = borders[i, :length]
            end_of_audio = length * ms_per_step

            b = predicted_border * ms_per_step

            switched = False
            prev = 0
            if duration_combined_model is not None:
                duration = duration_batch[i].reshape(-1)

            for i, v in enumerate(b):
                v = b[i]
                if v < prev:
                    switched = True
                    # prev + 0.01 # end of file? end_of_audio
                    after = b[i + 1] if i + 1 < len(b) else end_of_audio
                    if duration_combined_model is not None:
                        v = after - duration[i - 1]

                if v < prev:
                    v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001
                b[i] = v
                prev = v

            prev = 0
            for i, v in enumerate(b):
                assert v >= prev, f"This should never happen! {i}"
                prev = v

            diff = truth_border * ms_per_step - b
            # print("truth", (truth_border * ms_per_step).round().tolist())
            # print("gen  ", b.round().tolist())
            if np.abs(diff).max() > report_error:
                print(f"[id:{idx:3d}]  [{diff.min():5.0f} {diff.max():5.0f}]  {length:4d} {switched}")

            diffs.append(diff)
            label_ids.append(label_id)

    diff = np.concatenate(diffs)
    label_ids = np.concatenate(label_ids)

    phoneme_map = defaultdict(list)
    for pdur, pid in zip(diff, label_ids):
        phoneme_map[pid].append(abs(pdur))

    if plotting:
        for func in [np.max, np.mean]:
            print(func)
            mean_phoneme_dur = sorted(
                ([f"{KNOWN_LABELS[pid].ljust(4)}", func(val)] for pid, val in phoneme_map.items()),
                key=lambda x: x[1],
            )
            print(len(mean_phoneme_dur))
            for row in zip(
                mean_phoneme_dur[::5],
                mean_phoneme_dur[1::5],
                mean_phoneme_dur[2::5],
                mean_phoneme_dur[3::5],
                mean_phoneme_dur[4::5],
            ):
                for i, (p, c) in enumerate(row):
                    print(f"{p} {c:5.2f}ms".ljust(20), end="", sep="")
                print()
            for i, (p, c) in enumerate(mean_phoneme_dur[-4:]):
                print(f"{p} {c:5.2f}ms".ljust(20), end="", sep="")
            print("\n.")
            figure()
            plt.plot(*zip(*[[len(val), func(val)] for pid, val in phoneme_map.items()]), "wo")
            for pid, val in phoneme_map.items():
                plt.annotate(KNOWN_LABELS[pid], xy=(len(val), func(val)))
            plt.xlabel("Occurence count", fontsize=13)
            plt.ylabel("Mean error" if func is np.mean else "Max error", fontsize=13)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%dms"))
            plt.show()

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    torch.save(diff, "diff.pth")
    display_diff(diff, "position", plotting=plotting)


def explore_inherit_border_error(dataset):
    print("""[PLOT] How much do the borders of processed audio transcription differ from the original timesteps?""")
    time_iter = BucketIterator(
        dataset,
        batch_size=64,
        sort_key=lambda x: len(x.features),
        sort=False,
        shuffle=True,
        sort_within_batch=True,
    )

    diffs = []
    for batch in time_iter:
        batch_s = len(batch.label_vec)
        for i in range(batch_s):
            original = batch.label_vec[i].cpu().numpy()
            trans = batch.transcription[i].cpu().numpy()
            original_ids = np.argmax(original, axis=1)
            mapping = batch.out_map[i]

            try:
                _a, _b, d = find_borders(original_ids, mapping)
                diffs.append(d)
            except Exception as e:
                print(e)
                arr = (np.where(original_ids[:-1] != original_ids[1:])[0] + 0.5) * WIN_STEP * 1000
                last = original_ids.shape[0] * WIN_STEP * 1000
                arr = np.append(arr, last)
                print(
                    len(mapping),
                    len(dedupe(original_ids)),
                    arr.shape,
                    trans.shape,
                    original_ids[:10],
                    original_ids[-10:],
                )
                prev = 0
                for (voc, t), oid in zip(mapping, dedupe(original_ids) + [99, 99, 99]):
                    print(f"{int(voc)}-{int(oid)} \t{t:.0f}\t{t - prev:.0f}")
                    prev = t

    diff = np.concatenate(diffs)
    display_diff(diff, "trans. vs. timesteps", plotting=True)


def draw_counts(counts, name):
    print(f"[dataset rows]{name}: {len(counts)}")
    _, axarr = plt.subplots(1, 2, figsize=(10, 3))
    axarr[0].hist(counts, bins=25)
    axarr[1].hist(counts, bins=10)
