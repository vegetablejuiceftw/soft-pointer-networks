from constants import *


class AudioCaching:
    BASE_FOLDER = '/content/TIMIT-PLUS'
    CACHE_FOLDER = BASE_FOLDER + '-CACHE'
    AUDIO_RATE = 16000

    @classmethod
    def to_file_path(cls, file_path, key):
        filename = os.path.basename(file_path)
        cache_filename = f"[{'__'.join(str(k) for k in key)}]{filename}"
        cache_path = os.path.dirname(file_path).replace(cls.BASE_FOLDER, cls.CACHE_FOLDER)
        cache_file_path = os.path.join(cache_path, cache_filename)
        return cache_file_path

    @classmethod
    def set(cls, file_path, key, audio):
        cache_file_path = cls.to_file_path(file_path, key)
        # cached version exists :D
        if not os.path.isfile(cache_file_path):
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            sf.write(cache_file_path, audio, cls.AUDIO_RATE)

    @classmethod
    def get(cls, file_path, key):
        cache_file_path = cls.to_file_path(file_path, key)
        # cached version exists :D
        if os.path.isfile(cache_file_path):
            print("+", cache_file_path)
            return cls.load(cache_file_path)
        else:
            print("-", file_path)

    @classmethod
    def load(cls, file_path):
        audio = sf.read(file_path)[0]
        return audio


class UtteranceBatch(NamedTuple):
    padded: torch.tensor
    masks: torch.tensor
    lengths: torch.tensor


class Utterance(NamedTuple):
    features: torch.tensor
    labels: torch.tensor
    transcription: torch.tensor
    label_vec: torch.tensor
    out_map: List
    out_duration: torch.tensor
    in_transcription: torch.tensor
    position: torch.tensor
    border: torch.tensor
    weight: torch.tensor

    index: int
    key: str
    audio_file: str
    label_file: str


def stack(arr, tensor):
    return [tensor(a).to(device) for a in arr]


def dedupe(tags):
    labels = []
    last = None
    for x in tags:
        if last is not None and last == x:
            continue
        last = x
        labels.append(x)
    return labels


def find_borders(output_ids, original_mapping):
    # add half to the border, as it is between the two frames
    arr = (np.where(output_ids[:-1] != output_ids[1:])[0] + 0.5) * ms_per_step
    last = output_ids.shape[0] * ms_per_step
    arr = np.append(arr, last)
    flat_ids = np.array(dedupe(output_ids))
    ids = np.array([voc for voc, end in original_mapping])
    assert (ids.shape == flat_ids.shape) and (
            ids != flat_ids).sum() == 0, f"[error] Mapping and Output have same composition {ids.shape} {flat_ids.shape}"
    a, b = np.array([end for voc, end in original_mapping]), arr

    diff = (a - b)
    return a, b, diff


class PositionalEncodingLabeler(nn.Module):
    def __init__(self, d_model, dropout=0.1, scale=1, max_len=2048):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)
        if not self.scale:
            return
        max_len = int(max_len * scale)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        m = nn.Upsample(scale_factor=(1. / scale, 1), mode='bilinear', align_corners=True)
        shape = pe.shape
        pe = pe.view(1, 1, *shape)
        pe = m(pe).view(-1, d_model)

        pe = pe.unsqueeze(0).transpose(0, 1).transpose(0, 1)[0]
        self.register_buffer('pe', pe)

    def forward(self, out_duration):
        durations = torch.cumsum(out_duration, dim=0) * DURATION_SCALER / ms_per_step
        return self.pe[durations.round().long()], durations


class DirectMaskDataset(Dataset):
    base = '/content'
    CACHE = {}

    @classmethod
    def load_csv(cls, prefix, sa=False):
        file_path = join(cls.base, f'{prefix}_data.csv')
        """ Filters out the files names of phonetic and sound data in pairs"""
        df = pd.read_csv(file_path, delimiter=',', nrows=None)
        df = df.sort_values(by=['path_from_data_dir'])
        # audio_mask = df.is_converted_audio == True
        audio_mask = (df.is_audio == True) & (df.is_converted_audio == True)
        phn_mask = df.filename.str.contains('.PHN') == True
        SA_mask = df.filename.str.contains('SA') == False
        df = df.loc[audio_mask | phn_mask]
        print("SA", sa)
        if not sa:
            df = df.loc[SA_mask]

        ipd.display(df.head())
        nRow, nCol = df.shape
        # print(f'There are {nRow} rows and {nCol} columns')
        A, B = df.loc[phn_mask].path_from_data_dir, df.loc[audio_mask].path_from_data_dir
        assert len(A) == len(B)
        files = list(zip(A, B))
        return files

    @staticmethod
    def get_name(file_name):
        _, a, b, name = file_name.split('/')
        name = name.split('.')[0]
        return a, b, name

    def json_to_vec(arr):
        return [np.array([MAP_LABELS[tag][0] for tag in tags]) for tags in arr]

    def process_audio(self, labels: list, length: int, step: float) -> List[List]:
        tag_ints, tag_vecs, tag_mapping, transcription = [], [], [], []

        current, prev = 0, None
        for end_ms, tag in labels:

            tag = TRANSFORM_MAPPING.get(tag, tag)
            q_tag = False  # /q/ tag
            if tag is None:
                tag = prev
                q_tag = True
                end_ms = (current + end_ms) / 2

            if current >= end_ms:
                continue

            unholy_combination = prev in NO_BORDER_MAPPING and tag in NO_BORDER_MAPPING
            # if unholy_combination:
                # unholy.add((prev, tag))

            if prev == tag and MERGE_DOUBLES or unholy_combination or q_tag:
                tag_id, ems = tag_mapping[-1]
                tag_mapping[-1] = (tag_id, end_ms)

            else:
                tag_id = MAP_LABELS[tag][1]
                tag_vec = np.array(MAP_LABELS[tag][0])

                tag_mapping.append((tag_id, end_ms))
                transcription.append(tag_vec)

            prev = tag  # handle same tag occurence

            tag_ints.append(tag_id)
            tag_vecs.append(tag_vec)
            current += step

            while current < end_ms and len(tag_ints) < length:
                tag_ints.append(tag_id)
                tag_vecs.append(tag_vec)
                current += step

        if length > len(tag_vecs):
            tag_ints.append(tag_id)
            tag_vecs.append(tag_vec)

        return tag_ints, tag_vecs, tag_mapping, transcription

    @classmethod
    def get_set(cls, key, func):
        value = cls.CACHE.get(key)
        value = value if value is not None else func()
        cls.CACHE[key] = value
        return value

    def __init__(self, files, limit=None, mask=None, augment=False, duplicate=1, seed="42"):
        random = Random(seed)  # init random generator
        pos_prep = PositionalEncodingLabeler(POS_DIM, scale=POS_SCALE)

        self.counts = []

        base = self.base

        if limit is not None:
            files = files[:limit]

        inp, out_vec, out_int, out_map, out_dur, out_trans = [], [], [], [], [], []
        position, border, weight = [], [], []

        duplicate_set = set()
        self.files = []

        for i, (label_file, audio_file) in enumerate(files * duplicate):
            assert self.get_name(label_file) == self.get_name(audio_file)
            a, b, c = self.get_name(label_file)
            identifier = f'{a}_{b}_{c}_{i}'

            label_file = os.path.join(base, 'data', label_file)
            audio_file = os.path.join(base, 'data', audio_file)

            loader = lambda: AudioCaching.load(audio_file)
            audio = self.get_set(audio_file, loader)
            audio_scaling, rate = 32768. / 512, 16000
            audio_base_len = len(audio)

            stretch = 1
            pure_key = (audio_file, "pure_key")
            if pure_key not in duplicate_set:
                duplicate_set.add(pure_key)
            elif augment:
                # pitch = random.choice([-6, -4, -1, 1, 4, 6])
                # stretch = random.choice([0.85, 0.9, 0.95, 1.05, 1.1, 1.15])
                # pitch = random.choice([-4, -1, 1, 4])
                pitch = random.choice([-1, 0, 1])
                stretch = random.choice([0.9, 0.95, 1.05, 1.1])

                key_stretch = "time_stretch", stretch
                key_pitch = "pitch_shift", pitch, stretch

                duplication_key = (audio_file, key_pitch)
                if duplication_key in duplicate_set:
                    continue
                duplicate_set.add(duplication_key)

                audio_pitch_shift = lambda: pyrb.pitch_shift(audio, rate, pitch)

                cache_audio = AudioCaching.get(audio_file, key_pitch)
                got_pitch = cache_audio is not None
                cache_audio = cache_audio if got_pitch else AudioCaching.get(audio_file, key_stretch)
                got_stretch = cache_audio is not None

                audio = cache_audio if cache_audio is not None else audio

                if not got_stretch:
                    audio = pyrb.time_stretch(audio, rate, stretch)
                    AudioCaching.set(audio_file, key_stretch, audio)

                if not got_pitch:
                    audio = pyrb.pitch_shift(audio, rate, pitch)
                    AudioCaching.set(audio_file, key_pitch, audio)

                stretch = len(audio) / audio_base_len

            fbank_feat = logfbank(audio, rate, winlen=WIN_SIZE, winstep=WIN_STEP,
                                  nfilt=INPUT_SIZE)  # TODO: remove scaling
            # some audio instances are too short for the audio transcription and the winlen cut :(
            fbank_feat = np.vstack([fbank_feat] + [fbank_feat[-1]] * 10)

            step_size = (WIN_STEP * 1000)
            with open(label_file) as f:
                lines = list(f.readlines())
                length = fbank_feat.shape[0]
                length_ms = length * step_size
                labels = []
                ms_samples = 16

                for line in lines:
                    _, end, tag = line.split()
                    end_ms = float(end) / ms_samples * stretch
                    end_ms = min(end_ms, length_ms)
                    labels.append((end_ms, tag))

                length = int((end_ms / step_size))

            tag_ints, tag_vecs, tag_mapping, transcription = self.process_audio(labels, length, step_size)
            fbank_feat = fbank_feat[:len(tag_ints)]
            length = fbank_feat.shape[0]
            length_ms = length * step_size

            w = [200. / FOUND_LABELS[KNOWN_LABELS[_pid]] for _pid, _ms in tag_mapping]

            if i % 150 == 0:
                print(i)
                gc.collect()

            if length == len(tag_vecs) and length == len(tag_ints):
                original = stack([tag_vecs], torch.FloatTensor)[0].cpu().numpy()
                original_ids = np.argmax(original, axis=1)
                if MERGE_DOUBLES:
                    a, b, diff = find_borders(original_ids, tag_mapping)
                    d = abs(diff).max()
                    if d > 15:
                        print(f"[DIFF-ERROR] diff is bigger {d} > 15", np.where(abs(diff) > 15), diff.shape)
                        print("\t", tag_mapping[-1], length_ms)
                        print("\t", np.round(a[-5:], 0))
                        print("\t", np.round(b[-5:], 0))
                        continue
                self.counts.append(length)
                tag_duration = []
                start = 0
                for _, end_ms in tag_mapping:
                    end_time = end_ms / DURATION_SCALER
                    tag_duration.append(end_time - start)
                    start = end_time  # CUMSUM vs DURATION

                pos, bor = pos_prep(torch.FloatTensor(tag_duration[:-1]).to(device))
                position.append(pos)
                border.append(bor)
                weight.append(w)

                out_dur.append(tag_duration)
                inp.append(fbank_feat)
                out_vec.append(tag_vecs)
                out_int.append(tag_ints)
                out_trans.append(transcription)
                out_map.append(tag_mapping)
                self.files.append((label_file, audio_file))
            else:
                print(
                    f"[ERROR] len not match {length} != {len(tag_vecs)} != {len(tag_ints)} \n\t - {label_file}\n\t - {audio_file}")

        self.inp = stack(inp, torch.FloatTensor)
        self.out_vec = stack(out_vec, torch.FloatTensor)
        self.out_int = stack(out_int, torch.LongTensor)
        self.transcription = stack(out_trans, torch.FloatTensor)
        self.out_map = out_map
        self.out_duration = stack(out_dur, torch.FloatTensor)
        self.in_transcription = stack(out_trans, torch.FloatTensor)
        self.key = [uuid.uuid4().urn for i in range(len(inp))]
        self.position = position
        self.border = border
        self.weight = stack(weight, torch.FloatTensor)

        FEATURES = RawField(postprocessing=self.features_batch_process)
        LABEL = RawField(postprocessing=self.features_batch_process)
        TRANSCRIPTION = RawField()
        LABEL_VEC = RawField()
        OUT_MAP = RawField()
        OUT_DUR = RawField(postprocessing=self.features_batch_process)
        IN_TRANS = RawField(postprocessing=self.features_batch_process)
        INDEX = RawField()
        KEY = RawField()
        POSITION = RawField(postprocessing=self.features_batch_process)
        BORDER = RawField(postprocessing=self.features_batch_process)
        WEIGHT = RawField(postprocessing=self.features_batch_process)

        setattr(FEATURES, "is_target", False)
        setattr(LABEL_VEC, "is_target", False)
        setattr(OUT_MAP, "is_target", False)
        setattr(TRANSCRIPTION, "is_target", False)
        setattr(OUT_DUR, "is_target", False)
        setattr(IN_TRANS, "is_target", False)
        setattr(LABEL, "is_target", True)
        setattr(INDEX, "is_target", False)
        setattr(KEY, "is_target", False)
        setattr(POSITION, "is_target", False)
        setattr(BORDER, "is_target", False)
        setattr(WEIGHT, "is_target", False)

        self.fields = {
            "features": FEATURES,
            "labels": LABEL,
            "transcription": TRANSCRIPTION,
            "label_vec": LABEL_VEC,
            "out_map": OUT_MAP,
            "out_duration": OUT_DUR,
            "in_transcription": IN_TRANS,
            "index": INDEX,
            "key": KEY,
            "position": POSITION,
            "border": BORDER,
            "weight": WEIGHT,
        }

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx) -> Utterance:
        label_file, audio_file = self.files[idx]
        return Utterance(
            self.inp[idx], self.out_int[idx], self.transcription[idx], self.out_vec[idx],
            self.out_map[idx],
            self.out_duration[idx], self.in_transcription[idx],
            self.position[idx], self.border[idx], self.weight[idx],
            idx, self.key[idx],
            label_file, audio_file,
        )

    @staticmethod
    def features_batch_process(batch) -> UtteranceBatch:
        # this is used when a list of data items is transformed into a batch
        # TODO: could we, should we use pack_padded_sequence
        padded = nn.utils.rnn.pad_sequence(batch, batch_first=True).to(device)
        lens = torch.tensor([len(item) for item in batch]).to(device)
        b, max_len, *f = padded.shape
        return UtteranceBatch(
            padded,
            torch.arange(max_len).expand(len(lens), max_len).to(device) < lens.unsqueeze(1),
            lens
        )

    def batch(self, batch_size=128, sort_key=lambda x: len(x.features), sort=False, shuffle=True,
              sort_within_batch=True):
        return BucketIterator(self, batch_size=batch_size, sort_key=sort_key, sort=sort, shuffle=shuffle,
                              sort_within_batch=sort_within_batch)
