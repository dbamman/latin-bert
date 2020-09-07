python3 scripts/generate_tagset.py UD_Latin-PROIEL/*conllu > UD_Latin-PROIEL/pos.tagset
python3 scripts/generate_tagset.py UD_Latin-Perseus/*conllu > UD_Latin-Perseus/pos.tagset
python3 scripts/generate_tagset.py UD_Latin-ITTB/*conllu > UD_Latin-ITTB/pos.tagset


# Train/test on Perseus:

python3 scripts/latin_static_pos_tagging.py -m train --max_epochs 25 -e ../../models/latin.200.vectors.txt  --trainFile UD_Latin-Perseus/la_perseus-ud-train.conllu --modelFile UD_Latin-Perseus/pos.static.model --tagFile UD_Latin-Perseus/pos.tagset > logs/static_perseus.train.log 2>&1

python3 scripts/latin_static_pos_tagging.py -m test -e ../../models/latin.200.vectors.txt  --testFile UD_Latin-Perseus/la_perseus-ud-test.conllu  --modelFile UD_Latin-Perseus/pos.static.model --tagFile UD_Latin-Perseus/pos.tagset > logs/static_perseus.test.log 2>&1

# Train/test on ITTB:

python3 scripts/latin_static_pos_tagging.py -m train -e ../../models/latin.200.vectors.txt  --trainFile UD_Latin-PROIEL/la_proiel-ud-train.conllu --devFile UD_Latin-PROIEL/la_proiel-ud-dev.conllu --modelFile UD_Latin-PROIEL/pos.static.model --tagFile UD_Latin-PROIEL/pos.tagset > logs/static_proiel.train.log 2>&1

python3 scripts/latin_static_pos_tagging.py -m test -e ../../models/latin.200.vectors.txt  --testFile UD_Latin-PROIEL/la_proiel-ud-test.conllu  --modelFile UD_Latin-PROIEL/pos.static.model --tagFile UD_Latin-PROIEL/pos.tagset > logs/static_proiel.test.log 2>&1

# Train/test on PROIEL:

python3 scripts/latin_static_pos_tagging.py -m train -e ../../models/latin.200.vectors.txt  --trainFile UD_Latin-ITTB/la_ittb-ud-train.conllu --devFile UD_Latin-ITTB/la_ittb-ud-dev.conllu --modelFile UD_Latin-ITTB/pos.static.model --tagFile UD_Latin-ITTB/pos.tagset > logs/static_ittb.train.log 2>&1

python3 scripts/latin_static_pos_tagging.py -m test -e ../../models/latin.200.vectors.txt  --testFile UD_Latin-ITTB/la_ittb-ud-test.conllu  --modelFile UD_Latin-ITTB/pos.static.model --tagFile UD_Latin-ITTB/pos.tagset > logs/static_ittb.test.log 2>&1
 
