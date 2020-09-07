# latin-bert/case\_studies/infilling

Carry out the following the reproduce results from the infilling experiments in Bamman and Burns (2020); results are saved to the logs/ directory.

```sh
python3 scripts/predict_word.py --bertPath ../../models/latin_bert/ --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder -d data/emendation_filtered.txt > logs/infilling.log 2>&1
```

Note:

`data/emendation_filtered.txt` contains the evaluation data for text emendations; `scripts/create_data` documents the process of its creation.