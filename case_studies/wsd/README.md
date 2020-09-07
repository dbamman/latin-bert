# latin-bert/case\_studies/wsd

Carry out the following the reproduce results from the WSD experiments in Bamman and Burns (2020); results are saved to the logs/ directory.

Get Lewis and Short from Perseus Github:

```sh
wget --no-check-certificate https://github.com/PerseusDL/lexica/raw/master/CTS_XML_TEI/perseus/pdllex/lat/ls/lat.ls.perseus-eng1.xml 
mv lat.ls.perseus-eng1.xml data/
```

Extract sense data to create WSD training/evaluation data:

```sh
python3 scripts/parse_ls.py data/lat.ls.perseus-eng1.xml > data/senses.txt
python3 scripts/create_wsd_data.py data/senses.txt data/latin.lemmas.txt > data/latin.sense.data
```

Train and evaluate BERT model:

```sh
python3 scripts/latin_wsd_bert.py -m train --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder -f data/wsd_bert.model --max_epochs 100 -i data/latin.sense.data > logs/wsd_bert.log 2>&1
```

Train and evaluate static embeddings model:

```sh
python3 scripts/latin_wsd_static.py -m train -f data/wsd_static.model --max_epochs 100 -i data/latin.sense.data -e ../../models/latin.200.vectors.txt > logs/wsd_static.log 2>&1
```
