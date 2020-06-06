# CoNLL paper experiments

1. Generate all models in numpy format
2. Reduce models (align their vocabularies)
3. Convert reduced models to text files

## Transorm numpy models with Conceptor-Negation
```python
import os
import sys
import numpy as np


def apply_conceptor_negation(vectors, alpha):
    num_words = vectors.shape[0]
    dim = vectors.shape[1]
    x_collector = vectors.T  # put the word vectors in columns
    R = x_collector.dot(x_collector.T) / num_words  # calculate the un-centered correlation matrix
    C = R @ np.linalg.inv(R + alpha ** (-2) * np.eye(dim))  # calculate the conceptor matrix
    cn_vectors = ((np.eye(dim) - C) @ x_collector).T
    return cn_vectors


if __name__ == '__main__':
    model_filepath = sys.argv[1]
    output_filepath = os.path.join(os.path.dirname(
        os.path.abspath(sys.argv[1])), 'cn.{}'.format(
            os.path.basename(sys.argv[1])))
    vectors = np.load(model_filepath)

    print('Post-processing word vectors with CN...')
    cn_vectors = apply_conceptor_negation(vectors, alpha=2)
    print('Done postprocessing word vectors')

    print('Saving numpy postprocessed vectors...')
    np.save(output_filepath, cn_vectors)
    print('Done saving numpy vectors')
```

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

import embeddix.core.evaluator as evaluator

if __name__ == '__main__':
    MODELS_DIRPATH = '/home/debian/embeddings/reduced/'
    output_filename = 'results-{}.tsv'.format(datetime.datetime.now().timestamp())
    models_filepaths = [os.path.join(MODELS_DIRPATH, filename) for filename in
                        os.listdir(MODELS_DIRPATH) if filename.endswith('.npy')]
    with open(output_filename, 'w', encoding='utf-8') as output_stream:
        print('NAME\tMEN-SPR\tSIMLEX-SPR\tSIMVERB-SPR\tAP\tBATTIG\tESSLI',
              file=output_stream)
        for model_filepath in tqdm(sorted(models_filepaths)):
            vocab_filepath = '{}.vocab'.format(model_filepath.split('.npy')[0])
            name = os.path.basename(model_filepath).split('.npy')[0]
            men_spr = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='men')
            simlex_spr = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='simlex')
            simverb_spr = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='simverb')
            ap = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='ap')
            battig = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='battig')
            essli = evaluator.evaluate_distributional_space(
                model_filepath, vocab_filepath, dataset='essli')
            # dim = model.shape[1]
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                name, men_spr, simlex_spr, simverb_spr, ap, battig, essli),
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
