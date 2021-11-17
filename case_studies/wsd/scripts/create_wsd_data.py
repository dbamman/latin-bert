import sys, re
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from unidecode import unidecode

word_tokenizer = WordTokenizer()

def read_lemmas(filename):
	lemmadict={}
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			word=cols[0].lower()
			lemma=re.sub("[^A-Za-z]", "", cols[1].lower())
			if word not in lemmadict:
				lemmadict[word]={}
			lemmadict[word][lemma]=1

	return lemmadict

def proc(filename, lemmadict):
	minLength=5
	minCount=10
	data={}
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			lemma=cols[1]
			senseid=cols[2]

			# just consider first two senses
			if senseid != "I" and senseid != "II":
				continue

			targetLemma=re.sub("\d", "", lemma)
			text=cols[5].lower()
			# remove trailing comma
			text=re.sub(",$", "", text)

			# make sure the quotation is mostly latin (not english)
			testtext=re.sub("[^A-Za-z ]", "", text).split(" ")
			latinwords=0
			for word in testtext:
				if word in lemmadict:
					latinwords+=1
			latinratio=float(latinwords)/len(testtext)

			if latinratio < .9:
				continue

			text = unidecode(text)
			tokens=word_tokenizer.tokenize(text)

			if len(tokens) >= minLength:
				if lemma not in data:
					data[lemma]={}
				if senseid not in data[lemma]:
					data[lemma][senseid]=[]
				before=[]
				after=[]
				seen=False
				targetWord=None
				for word in tokens:
					if word in lemmadict:
						lemmas=lemmadict[word]
					else:
						lemmas={word:1}

					if targetLemma in lemmas:
						targetWord=word
						seen=True
					else:
						if seen == False:
							before.append(word)
						else:
							after.append(word)
				if targetWord is not None:
					data[lemma][senseid].append((' '.join(before), targetWord, ' '.join(after)))

	for lemma in data:
		min_attestations=100000
		min_sense_counts=0
		for senseid in data[lemma]:
			count=len(data[lemma][senseid])

			if count > minCount:
				min_sense_counts+=1
				if count < min_attestations:
					min_attestations=count

		if min_sense_counts >= 2:
			for senseid in data[lemma]:
				count=len(data[lemma][senseid])
				if count > minCount:
					for before, target, after in data[lemma][senseid][:min_attestations]:
						print("%s\t%s\t%s\t%s\t%s" % (lemma, senseid, before, target, after))


lemmadict=read_lemmas(sys.argv[2])
proc(sys.argv[1], lemmadict)