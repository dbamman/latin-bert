python3 scripts/generate_tagset.py UD_Latin-PROIEL/*conllu > UD_Latin-PROIEL/pos.tagset
python3 scripts/generate_tagset.py UD_Latin-Perseus/*conllu > UD_Latin-Perseus/pos.tagset
python3 scripts/generate_tagset.py UD_Latin-ITTB/*conllu > UD_Latin-ITTB/pos.tagset

# Train/test on Perseus:
python3 scripts/latin_sequence_labeling.py -m train --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --trainFile UD_Latin-Perseus/la_perseus-ud-train.conllu --tagFile UD_Latin-Perseus/pos.tagset --modelFile UD_Latin-Perseus/pos.model --metric accuracy --max_epochs 5 > logs/bert_perseus.train.log 2>&1

python3 scripts/latin_sequence_labeling.py -m test --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --testFile UD_Latin-Perseus/la_perseus-ud-test.conllu --tagFile UD_Latin-Perseus/pos.tagset --modelFile UD_Latin-Perseus/pos.model --metric accuracy > logs/bert_perseus.test.log 2>&1

# Train/test on ITTB:
python3 scripts/latin_sequence_labeling.py -m train --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --trainFile UD_Latin-ITTB/la_ittb-ud-train.conllu --devFile UD_Latin-ITTB/la_ittb-ud-dev.conllu --tagFile UD_Latin-ITTB/pos.tagset --modelFile UD_Latin-ITTB/pos.model --metric accuracy > logs/bert_ittb.train.log 2>&1

python3 scripts/latin_sequence_labeling.py -m test --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --testFile UD_Latin-ITTB/la_ittb-ud-test.conllu --tagFile UD_Latin-ITTB/pos.tagset --modelFile UD_Latin-ITTB/pos.model --metric accuracy > logs/bert_ittb.test.log 2>&1

# Train/test on PROIEL:
python3 scripts/latin_sequence_labeling.py -m train --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --trainFile UD_Latin-PROIEL/la_proiel-ud-train.conllu --devFile UD_Latin-PROIEL/la_proiel-ud-dev.conllu --tagFile UD_Latin-PROIEL/pos.tagset --modelFile UD_Latin-PROIEL/pos.model --metric accuracy > logs/bert_proiel.train.log 2>&1

python3 scripts/latin_sequence_labeling.py -m test --bertPath ../../models/latin_bert --tokenizerPath ../../models/subword_tokenizer_latin/latin.subword.encoder --testFile UD_Latin-PROIEL/la_proiel-ud-test.conllu --tagFile UD_Latin-PROIEL/pos.tagset --modelFile UD_Latin-PROIEL/pos.model --metric accuracy > logs/bert_proiel.test.log 2>&1

