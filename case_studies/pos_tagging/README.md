# latin-bert/case\_studies/pos_tagging

Carry out the following the reproduce results from the POS tagging experiments in Bamman and Burns (2020); results are saved to the logs/ directory.

### Get UD Latin POS data

```sh
git clone https://github.com/UniversalDependencies/UD_Latin-Perseus.git
git clone https://github.com/UniversalDependencies/UD_Latin-ITTB.git
git clone https://github.com/UniversalDependencies/UD_Latin-PROIEL.git
```

### Run evaluation with Latin BERT

```sh
./scripts/run_bert_eval.sh
```

### Run evaluation with static word2vec embeddings:

```sh
./scripts/download_static_vectors.sh
./scripts/run_static_eval.sh
```