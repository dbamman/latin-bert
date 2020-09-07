# download all training data for Latin BERT
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xi-pbW9CbzMdaaJOOuj3JW0oMt3EuNDb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xi-pbW9CbzMdaaJOOuj3JW0oMt3EuNDb" -O latin_bert_training_data.txt.gz && rm -f /tmp/cookies.txt
gunzip latin_bert_training_data.txt.gz
mv latin_bert_training_data.txt data/

# make sure emendation data didn't show up in training data
python3 scripts/check_dupes_emendations.py data/emendation_context.csv data/latin_bert_training_data.txt > data/emendation_filtered.txt
