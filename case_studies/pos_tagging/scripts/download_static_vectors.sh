# Download static word2vec embeddings
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zhpyg5vxMT0iSMl7iW7KLW2wZ7phZFKE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zhpyg5vxMT0iSMl7iW7KLW2wZ7phZFKE" -O latin.200.vectors.txt && rm -f /tmp/cookies.txt
mv latin.200.vectors.txt ../../models/



