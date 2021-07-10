# Dependencies

1. To get started, you will need to install version 0.6.2 of the HuggingFace transformer repo. I've included a copy [here](../code/transformer_version/pytorch-pretrained-BERT-master.zip). Unpack the directory and install, as with

```bash
pip install --editable .
```

2. The main code has been most recently tested with the following versions of the additional required dependencies:
  * Python 3.7.10
  * torch 1.8.1
  * gensim '3.8.0'
  * sklearn '0.24.1'
  * numpy '1.20.2'

3. For applicable experiments, the pre-trained Word2Vec word embeddings of Mikolov et al. (2013) can be downloaded from [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/), which has a link to a compressed copy of the file (`GoogleNews-vectors-negative300.bin.gz`). Similarly, the GloVe embeddings (Pennington, Socher, and Manning 2014), used for a subset of experiments for comparison to certain previous models, are available here: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). The GloVe embeddings need to be converted to word2vec format using gensim (see above dependencies), as follows:

```
python -m gensim.scripts.glove2word2vec --input glove.6B.300d.txt --output glove.6B.300d.txt.word2vec_format.txt
```

The file `glove.6B.300d.txt.word2vec_format.txt` can then be used during training as the argument to `--word_embeddings_file` when used in conjunction with the `--word_embeddings_file_in_plaintext` flag.

# Acknowledgements

This repo incorporates, references, and/or compares to code and/or data from the following public repos. If you find this repo useful, like/star their repos, as well.
  * The HuggingFace Transformer repo, the most recent iteration of which is here: https://github.com/huggingface/transformers
  * Taeuk Kim's PyTorch reimplementation (a version downloaded circa summer 2018) of Yoon Kim's 2014 paper ["Convolutional Neural Networks for Sentence Classification"](https://www.aclweb.org/anthology/D14-1181.pdf): https://github.com/galsang/CNN-sentence-classification-pytorch
  * The data of Kaushik, Hovy, and Lipton (2019): https://github.com/allenai/contrast-sets
  * The data of Gardner et al. (2020): https://github.com/acmi-lab/counterfactually-augmented-data
  * The code of Rei and Søgaard (2018): https://github.com/marekrei/mltagger
  * The code of Bujel, Yannakoudakis, and Rei (2021): https://github.com/bujol12/bert-seq-interpretability
