import sys, re
from cltk.tokenize.word import WordTokenizer
from cltk.tokenize.latin.sentence import SentenceTokenizer

word_tokenizer = WordTokenizer('latin')

def filt(text):
	text=re.sub("<", "", text)
	text=re.sub(">", "", text)
	return ' '.join(word_tokenizer.tokenize(text))

def strip_braces(text):
	text=re.sub("<", "", text)
	text=re.sub(">", "", text)
	return text


def read_test(filename):
	tests=[]

	with open(filename) as file:
		for line in file:
			data=re.sub(" +", " ", line.lower())
			cols=data.split("\t")

			left=word_tokenizer.tokenize(cols[3])
			target=word_tokenizer.tokenize(cols[4])
			if len(target) != 1:
				continue
			right=word_tokenizer.tokenize(cols[5].rstrip())
			if len(left) < 2 or len(right) < 2:
				continue

			target=target[0]
			origq="%s <%s> %s" % (' '.join(left[-2:]), target, ' '.join(right[:2]))
			query=re.sub("\s", "", origq)
			query=query.lower()

			tests.append((query, origq, filt(cols[3]), filt(cols[4]), filt(cols[5].rstrip())))
	return tests

def search(filename, tests):
	with open(filename) as file:
		data=file.read()
		data=re.sub("\s", "", data)
		data=data.lower()

		for (test, origq, left, target, right) in tests:
			matcher=re.search(re.escape(test), data)
			if matcher is None:
				stripped_test=strip_braces(test)
				matcher2=re.search(re.escape(stripped_test), data)
				if matcher2 is None:
					print("disjoint\t%s\t%s\t%s\t%s" % (origq, left, target, right))
				else:
					print("dupe\t%s\t%s\t%s\t%s" % (origq, left, target, right))

			else:
				print("dupe\t%s\t%s\t%s\t%s" % (origq, left, target, right))

tests=read_test(sys.argv[1])
search(sys.argv[2], tests)