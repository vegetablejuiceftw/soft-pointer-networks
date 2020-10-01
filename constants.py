from os.path import join

BASE_PATH = "/content/TIMIT-PLUS"
FULL_FODLER_PATH = join(BASE_PATH, "data")

# WIN_STEP = 0.015
WIN_STEP = 0.010
# WIN_STEP = 0.005

WIN_SIZE = 0.025
# WIN_SIZE = 0.030

# duration models do not play well with 123ms bot not also with 0.1sec ...
DURATION_SCALER = 256.0

INPUT_SIZE = 26
ms_per_step = WIN_STEP * 1000
POS_DIM = 32
POS_SCALE = 1
MERGE_DOUBLES = False

FOUND_LABELS = dict(
    [
        ("h#", 12600),
        ("ix", 11587),
        ("s", 10114),
        ("iy", 9663),
        ("n", 9569),
        ("r", 9064),
        ("tcl", 8978),
        ("l", 8157),
        ("kcl", 7823),
        ("ih", 6760),
        ("dcl", 6585),
        ("k", 6488),
        ("t", 5899),
        ("m", 5429),
        ("ae", 5404),
        ("eh", 5293),
        ("z", 5046),
        ("ax", 4956),
        ("q", 4834),
        ("d", 4793),
        ("axr", 4790),
        ("w", 4379),
        ("aa", 4197),
        ("ao", 4096),
        ("dh", 3879),
        ("dx", 3649),
        ("pcl", 3609),
        ("p", 3545),
        ("ay", 3242),
        ("ah", 3185),
        ("f", 3128),
        ("ey", 3088),
        ("b", 3067),
        ("sh", 3034),
        ("gcl", 3031),
        ("ow", 2913),
        ("er", 2846),
        ("g", 2772),
        ("v", 2704),
        ("bcl", 2685),
        ("ux", 2488),
        ("y", 2349),
        ("epi", 2000),
        ("ng", 1744),
        ("jh", 1581),
        ("hv", 1523),
        ("pau", 1343),
        ("nx", 1331),
        ("hh", 1313),
        ("el", 1294),
        ("ch", 1081),
        ("th", 1018),
        ("en", 974),
        ("oy", 947),
        ("aw", 945),
        ("uh", 756),
        ("uw", 725),
        ("ax-h", 493),
        ("zh", 225),
        ("em", 171),
        ("eng", 43),
    ],
)

TRANSFORM_MAPPING = {
    # First, the sentence-beginning and sentence-ending pause symbols /h#/
    # were mapped to pause (/pau/).
    "h#": "pau",
    # Epenthetic silence (/epi/) was also mapped to pause.
    "epi": "pau",
    # The syllabic phonemes /em/, /en/, /eng/, and /el/ were mapped to their
    # non-syllabic counterparts /m/, /n/, /ng/, and /l/, respectively.
    "em": "m",
    "en": "n",
    "eng": "ng",
    "el": "l",
    # The glottal closure symbol /q/ was merged based on weird rules
    "q": None,
}

NO_BORDER_MAPPING = {
    "pau",
    "pcl",
    "bcl",
    "tcl",
    "dcl",
    "kcl",
    "gcl",
}

KNOWN_LABELS = list(
    sorted(set(TRANSFORM_MAPPING.get(k, k) for k in sorted(FOUND_LABELS.keys()) if TRANSFORM_MAPPING.get(k, k))),
)
KNOWN_LABELS_COUNT = len(KNOWN_LABELS)

MAP_LABELS = {
    label: ([int(KNOWN_LABELS.index(label) == i) for i in range(KNOWN_LABELS_COUNT)], KNOWN_LABELS.index(label))
    for label in KNOWN_LABELS
}
