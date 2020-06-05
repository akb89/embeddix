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

```python
import os
import datetime

from tqdm import tqdm

import entropix.utils.data as dutils
import entropix.core.evaluator as evaluator

if __name__ == '__main__':
    MODELS_DIRPATH = '/home/debian/embeddings/reduced/'
    output_filename = 'results-{}.tsv'.format(datetime.datetime.now().timestamp())
    models_filepaths = [os.path.join(MODELS_DIRPATH, filename) for filename in
                        os.listdir(MODELS_DIRPATH) if filename.endswith('.npy')]
    with open(output_filename, 'w', encoding='utf-8') as output_stream:
        print('NAME\tMEN-SPR\tSIMLEX-SPR\tSIMVERB-SPR\tAP\tBATTIG\tESSLI\tDIM',
              file=output_stream)
        for model_filepath in tqdm(sorted(models_filepaths)):
            vocab_filepath = '{}.vocab'.format(model_filepath.split('.npy')[0])
            model, vocab = dutils.load_model_and_vocab(
                model_filepath, 'numpy', vocab_filepath)
            name = os.path.basename(model_filepath).split('.npy')[0]
            men_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='men', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)[0]
            simlex_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='simlex', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)[0]
            simverb_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='simverb', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)[0]
            ap = evaluator.evaluate_distributional_space(
                model, vocab, dataset='ap', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            battig = evaluator.evaluate_distributional_space(
                model, vocab, dataset='battig', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            essli = evaluator.evaluate_distributional_space(
                model, vocab, dataset='essli', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            dim = model.shape[1]
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                name, men_spr, simlex_spr, simverb_spr, ap, battig, essli, dim),
                  file=output_stream)
```


## Run CoNLL00 CoNLL03 and PTB experiments

Clone `https://github.com/akb89/RepEval-2016`

Change all absolute filepath: `/home/debian/venv/bin`.

Manually install dependencies:
```shell
/home/debian/venv/bin/pip install keras==0.3.3
```

Then, run:
```shell
./ExtrinsicEva.sh /home/debian/embeddings/reduced/ /home/debian/RepEval-2016/results-reduced.txt
```

## Run Relation Extraction, Sentence polarity, sentiment analysis, SNLI and sentence subjectivity experiements

## Run RMSE computation

```python
import os

import itertools
import numpy as np

from tqdm import tqdm

import matrixor


if __name__ == '__main__':
    INPUT_DIRPATH = '/home/debian/embeddings/reduced/'
    MODELS_PATH = [os.path.join(INPUT_DIRPATH, filename) for filename in
                   os.listdir(INPUT_DIRPATH) if filename.endswith('.npy')]
    total_num = len([_ for _ in itertools.combinations(MODELS_PATH, 2)])
    for A_path, B_path in tqdm(itertools.combinations(MODELS_PATH, 2), total=total_num):
        A_name = os.path.basename(A_path).split('.npy')[0]
        B_name = os.path.basename(B_path).split('.npy')[0]
        A = np.load(A_path)
        B = np.load(B_path)
        rmse = matrixor.align(A, B)
        print('{}\t{}\tRMSE = {}'.format(A_name, B_name, rmse))

```
