import argparse
import copy, re
import sys
from transformers import BertModel, BertForMaskedLM, BertPreTrainedModel
from tensor2tensor.data_generators import text_encoder
import torch
import numpy as np
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from torch import nn
import random

random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder
		self.word_tokenizer = WordTokenizer()

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4

		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text):

		tokens=self.word_tokenizer.tokenize(text)

		wp_tokens=[]
		for token in tokens:

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:
				token=token.lower()
				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens

def proc(tokenids):

	mask_id=tokenids.index(wp_tokenizer.vocab["[MASK]"])
	torch_tokenids=torch.LongTensor(tokenids).unsqueeze(0)
	torch_tokenids=torch_tokenids.to(device)
	
	with torch.no_grad():
		preds = model(torch_tokenids)
		preds=preds[0]
		sortedVals=torch.argsort(preds[0][mask_id], descending=True)
		for k, p in enumerate(sortedVals[:1]):
			predicted_index=p.item()
			probs=nn.Softmax(dim=0)(preds[0][mask_id])
			return wp_tokenizer.reverseVocab[predicted_index]

def evaluate(wp_tokenizer, text_before_lacuna, text_after_lacuna, model, truth):

	tokens=[]
	tokens.extend(wp_tokenizer.tokenize(text_before_lacuna))
	position=len(tokens) + 1
	tokens.append("[MASK]")
	tokens.extend(wp_tokenizer.tokenize(text_after_lacuna))

	tokens.insert(0,"[CLS]")
	tokens.append("[SEP]")

	tokenids=wp_tokenizer.convert_tokens_to_ids(tokens)	

	mask_id=tokenids.index(wp_tokenizer.vocab["[MASK]"])

	torch_tokenids=torch.LongTensor(tokenids).unsqueeze(0)
	torch_tokenids=torch_tokenids.to(device)
	p1=0
	p10=0
	p50=0

	with torch.no_grad():
		preds = model(torch_tokenids)
		preds=preds[0]
		sortedVals=torch.argsort(preds[0][mask_id], descending=True)

		for k, p in enumerate(sortedVals[:50]):
			
			predicted_index=p.item()
			probs=nn.Softmax(dim=0)(preds[0][mask_id])


			suffix=""
			if not wp_tokenizer.reverseVocab[predicted_index].endswith("_"):
				uptokens=copy.deepcopy(tokenids)
				uptokens.insert(position, predicted_index)
				suffix=proc(uptokens)

			predicted_word="%s%s" % (wp_tokenizer.reverseVocab[predicted_index], suffix)
			predicted_word=re.sub("_$", "", predicted_word).lower()
			truth=truth.lower()
			if k == 0:
				print ("PRED\t%s\t%s\t%s\t%s\t%s\t%s" % (text_before_lacuna, "->", predicted_word, truth, "<-", text_after_lacuna))
			if k < 10:
				print("\t%s\t%.3f" % (predicted_word, probs[predicted_index]))
			if predicted_word == truth:
				if k == 0:
					p1=1
				if k < 10:
					p10=1
				if k < 50:
					p50=1
	
	return p1, p10, p50

def get_bucket(n):
	if n < 10:
		return 0
	elif n < 20:
		return 1
	elif n < 30:
		return 2
	elif n < 40:
		return 3
	elif n < 50:
		return 4
	elif n < 60:
		return 5
	elif n < 70:
		return 6
	elif n < 80:
		return 7
	elif n < 90:
		return 8
	elif n < 100:
		return 9

def proc_file(filename, wp_tokenizer, model):
	allp1=allp10=allp50=n=0

	minTokens=10
	maxTokens=100

	p1s=np.zeros(10)
	p10s=np.zeros(10)
	p50s=np.zeros(10)
	ns=np.zeros(10)

	with open(filename) as file:
		for idx, line in enumerate(file):

			cols=line.split("\t")
			if len(cols) < 5:
				continue

			cat=cols[0]
			if cat != "disjoint":
				continue
			text_before_lacuna=cols[2]
			truth=cols[3]

			# exclude punctuation and single characters from test
			if len(truth) < 2:
				continue
				
			text_after_lacuna=cols[4].rstrip()

			tot_toks=len(text_before_lacuna.split(" ")) + 1 + len(text_after_lacuna.split(" "))
				
			if tot_toks > minTokens and tot_toks < maxTokens:

				p1,p10,p50=evaluate(wp_tokenizer, text_before_lacuna, text_after_lacuna, model, truth)
				allp1+=p1
				allp10+=p10
				allp50+=p50

				bucket=get_bucket(tot_toks)
				p1s[bucket]+=p1
				p10s[bucket]+=p10
				p50s[bucket]+=p50
				ns[bucket]+=1
				
				n+=1

				if n % 10 == 0:
					print("Precision @ 1: %.3f 10: %.3f 50: %.3f, n: %s" % (allp1/n, allp10/n, allp50/n, n))
					sys.stdout.flush()


	for i in range(10):
		print("%s\t%.3f\t%.3f\t%.3f\t%s" % (i, p1s[i]/ns[i], p10s[i]/ns[i], p50s[i]/ns[i], ns[i]))

	print("Precision @ 1: %.3f 10: %.3f 50: %.3f, n: %s" % (allp1/n, allp10/n, allp50/n, n))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
	
	parser.add_argument('-f','--file', help='File containing data to get BERT representations for', required=False)
	parser.add_argument('-d','--datafile', help='Data file to evaluate', required=False)
	
	args = vars(parser.parse_args())

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]			
	dataFile=args["datafile"]			
	encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
	wp_tokenizer = LatinTokenizer(encoder)

	model = BertForMaskedLM.from_pretrained(bertPath)
	model.to(device)

	proc_file(dataFile, wp_tokenizer, model)
