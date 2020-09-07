# latin-bert/case\_studies/contextual\_nearest\_neighbors

Carry out the following the reproduce results from the contextual nearest neighbors experiments in Bamman and Burns (2020); results are saved to the logs/ directory.  (Note that this is both memory- and compute-intensive.)


Get Latin library texts:

```sh
./scripts/download.sh
```

Generate BERT representations for all texts (this takes about 10 hours on a GPU):

```sh
./scripts/run_bert_ll.sh
```

Start the interactive shell to query this corpus.  Note that this script reads in all BERT representations from the Latin Library and stores them in memory (it takes about an hour to read them all in with parallelization on a 10-core machine). The output of every query is displayed on screen and stored in the `output_comparisons` directory. 

```sh
python3 scripts/compare_bert_sents.py --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder -i data/latin_library_bert -o output_comparisons/
```

Queries take the form of \<query word\> \<query sentence\>, so if you want to search for representations similar to *amor* in *omnia vincit amor*, type the following:

```
> amor omnia vincit amor
```

To search for *in* in *Gallia est omnis divisa in partes tres*, type the following:

```
> in gallia est omnis divisa in partes tres
```

