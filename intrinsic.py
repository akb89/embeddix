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
