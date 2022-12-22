import os
from copy import deepcopy
from collections import Counter

from datasets import load_dataset

from dataloading import dto


def restructure(item: dict):
    item = deepcopy(item)
    item['source'] = item.pop('file').split('/data/')[-1].replace(".wav", "").replace(".WAV", "")
    item.update(item.pop('audio'))
    sr = item['sampling_rate']
    for timeline in item['phonetic_detail'], item['word_detail']:
        # timeline['start'] = np.array(timeline['start']) / sr
        # timeline['stop'] = np.array(timeline['stop']) / sr
        timeline['start'] = [e / (sr // 1000) for e in timeline['start']]
        timeline['stop'] = [e / (sr // 1000) for e in timeline['stop']]
    return item


# Speaker-Independent Phone Recognition Using Hidden Markov Models
# The syllabic phonemes /em/, /en/, /eng/, and /el/ were mapped to their
# non-syllabic counterparts /m/, /n/, /ng/, and /l/, respectively.
# five groups where within-group confusions are not counted:
# {sil , cl , vcl , epi} , {el, 1},
# {en, n} , {sh, zh} , {ao, aa} , {ih , ix} , {ah, ax} .
# https://www.researchgate.net/publication/247173709_Phoneme_Recognition_from_the_TIMIT_database_using_Recurrent_Error_Propagation_Networks
# https://cdn.intechopen.com/pdfs/15948/InTech-Phoneme_recognition_on_the_timit_database.pdf
# https://www.intechopen.com/chapters/15948
# this should result in 39 phonemes
TRANSFORM_MAPPING = {
    # First, the sentence-beginning and sentence-ending pause symbols /h#/
    # were mapped to pause (/pau/).
    "h#": "pau",
    # # Epenthetic silence (/epi/) was also mapped to pause.
    "epi": "pau",
    # The glottal closure symbol /q/ was merged based on weird rules
    "q": None,
    # folded
    "hv": "hh",
    "el": "l",
    "en": "n",
    "nx": "n",
    "zh": "sh",
    "ao": "aa",
    "ix": "ih",
    "ax": "ah",
    "ax-h": "ah",
    "em": "m",
    "ux": "uw",
    "axr": "er",
    "eng": "ng",
    "pcl": "pau",
    "tcl": "pau",
    "kcl": "pau",
    "qcl": "pau",
    "bcl": "pau",
    "dcl": "pau",
    "gcl": "pau",
}

count = 0


def fold_phonemes(item: dict):
    global count
    item = deepcopy(item)
    timeline = item['phonetic_detail']

    mapped = [
        (TRANSFORM_MAPPING.get(p) or p, p, stop)
        for p, stop in zip(
            timeline['utterance'],
            timeline['stop'],
        )
        if p != 'q'
    ]

    for i in range(len(mapped) - 1):
        p, old_p, stop = mapped[i]
        np, old_np, nstop = mapped[i + 1]

        if p == np and old_p != old_np and p != 'pau':
            # print(p, [old_p, old_np])
            count += 1
            mapped[i] = (None, None, None)

        if p == np and p == 'pau':
            # count += 1
            mapped[i] = (None, None, None)

    mapped = [
        (p, stop)
        for p, _, stop in mapped
        if p
    ]

    timeline['utterance'], timeline['stop'] = zip(*mapped)
    timeline['start'] = [0, *timeline['stop'][:-1]]
    return item


def produce(path: str):
    dataset = load_dataset("timit_asr", data_dir='.data')

    dataset_test = dto.apply(dataset['test'], restructure)
    dataset_train = dto.apply(dataset['train'], restructure)

    dataset_test = dto.apply(dataset_test, fold_phonemes)
    dataset_train = dto.apply(dataset_train, fold_phonemes)

    dto.write(dataset_test, os.path.join(path, "test_data.tar.xz"))
    dto.write(dataset_train, os.path.join(path, "train_data.tar.xz"))


def calculate_phoneme_counts(dataset, duration: int = None):
    phonemes = [
        p
        for item in dataset
        for p, d in zip(item.phonetic_detail.utterance, item.phonetic_detail.duration)
        if not duration or d < duration
    ]
    c = Counter(phonemes)
    print(len(c), "duration:", duration)
    print(dict(c.most_common()))


if __name__ == '__main__':
    produce('.data')
    print("done", count)
    result = dto.wds_load(".data/test_data.tar.xz") + dto.wds_load(".data/train_data.tar.xz")
    calculate_phoneme_counts(result)
    calculate_phoneme_counts(result, 16)

# TODO: DTW soft pointer network
# https://buckeyecorpus.osu.edu/
# The TIMIT “sa” sentences were considered unsuitable for training or testing as they consist ofonly two phrases and would introduce an unnatural bias in the distribution of ph onemes andtheir contexts.
# 2350 under 10 ms duration
# loudness normalize?
