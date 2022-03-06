import os.path

import gdown
from pkg_resources import ensure_directory


BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

output = os.path.join(BASE, ".data/data.zip")
if not os.path.exists(output):
    ensure_directory(output)
    out = gdown.download(id="15MxBckNzyEjO7cpY38O38NaWnssShl2l", output=output, quiet=False)
    gdown.extractall(out)
    print(out)
