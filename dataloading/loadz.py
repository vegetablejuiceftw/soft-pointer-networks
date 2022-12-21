# %%
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
        timeline['start'] = [e / sr for e in timeline['start']]
        timeline['stop'] = [e / sr for e in timeline['stop']]
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


def fold_phonemes(item: dict):
    item = deepcopy(item)
    timeline = item['phonetic_detail']
    mapped = [
        (TRANSFORM_MAPPING.get(p, p), start, stop)
        for p, start, stop in zip(
            timeline['utterance'],
            timeline['start'],
            timeline['stop'],
        )
        if p != 'q'
    ]
    timeline['utterance'], timeline['start'], timeline['stop'] = zip(*mapped)
    return item


dataset = load_dataset("timit_asr", data_dir='.data')
print(dataset)

dataset_test = dto.apply(dataset['test'], restructure)
dataset_train = dto.apply(dataset['train'], restructure)

dataset_test = dto.apply(dataset_test, fold_phonemes)
dataset_train = dto.apply(dataset_train, fold_phonemes)

# %%
# print(dataset_train[0])
phonemes = [p for item in dataset_test + dataset_train for p in item['phonetic_detail']['utterance']]
c = Counter(phonemes)
print(len(c))
print(dict(c.most_common()))


# %%
dto.write(dataset_test, ".data/test_data.tar.xz")
dto.write(dataset_train, ".data/train_data.tar.xz")

# %%
result = dto.wds_load(".data/train_data.tar.xz")
print(result[0].phonetic_detail.duration)

# TODO: DTW soft pointer network
