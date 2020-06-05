# CoNLL paper experiments

1. Generate all models in numpy format
2. Reduce models (align their vocabularies)
3. Convert reduced models to text files

## Reduce models
```shell
embeddix reduce --embeddings /home/debian/embeddings/original/
```

## Convert reduced models to text files
```python
import os
import numpy as np
from tqdm import tqdm

import embeddix

if __name__ == '__main__':
  INPUT_DIRPATH = '/home/debian/embeddings/reduced/'
  model_names = [filename.split('.npy')[0] for filename in
                   os.listdir(INPUT_DIRPATH) if filename.endswith('.npy')]
  for model_name in tqdm(model_names):
    model_filepath = os.path.join(INPUT_DIRPATH, '{}.npy'.format(model_name))
    vocab_filepath = os.path.join(INPUT_DIRPATH, '{}.vocab'.format(model_name))
    embeddix.convert_to_txt(model_filepath, vocab_filepath)
```

## Run MEN SIMLEX SIMVERB AP BATTIG ESSLI experiments


## Run CoNLL00 CoNLL03 and PTB experiments

Clone `https://github.com/akb89/RepEval-2016`

Change all absolute filepath: `/home/debian/venv/bin`.

Manually install dependencies:
```
/home/debian/venv/bin/pip install keras==0.3.3
```

Then, run:
```
./ExtrinsicEva.sh /home/debian/embeddings/reduced/ /home/debian/RepEval-2016/results-reduced.txt
```

## Run Relation Extraction, Sentence polarity, sentiment analysis, SNLI and sentence subjectivity experiements

## Run RMSE computation
