wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GRe3eFmQBDdF1kIT9T75aPTdquaf8Z8s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GRe3eFmQBDdF1kIT9T75aPTdquaf8Z8s" -O latin_library_text.tar.gz && rm -f /tmp/cookies.txt
mv latin_library_text.tar.gz data
cd data
gunzip latin_library_text.tar.gz
tar -xf latin_library_text.tar
rm latin_library_text.tar




