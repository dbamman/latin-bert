# Download BERT
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti' -O- | sed -rn 's/.*name="confirm" value="([0-9A-Za-z_]+)".*/\1\n/p')
UUID=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti' -O- | sed -rn 's/.*name="uuid" value="([0-9A-Za-z_-]+)".*/\1\n/p')
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?export=download&id=1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti&confirm=$CONFIRM&uuid=$UUID" -O latin_bert.tar && rm -f /tmp/cookies.txt
mv latin_bert.tar models/
cd models
tar -xf latin_bert.tar
rm latin_bert.tar



