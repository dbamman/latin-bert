import argparse, sys
from cltk.tokenize.word import WordTokenizer
from cltk.tokenize.latin.sentence import SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

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
		tokens=text.split(" ")
		wp_tokens=[]
		for token in tokens:

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens

def read_file(filename):
	sent_tokenizer = SentenceTokenizer()
	word_tokenizer = WordTokenizer('latin')

	all_sents=[]
	with open(filename, encoding="utf-8") as file:
		data=file.read()

		# BERT model is lowercase
		text=data.lower()

		sents=sent_tokenizer.tokenize(text)
		for sent in sents:
			tokens=word_tokenizer.tokenize(sent)
			filt_toks=[]
			for tok in tokens:
				if tok != "":
					filt_toks.append(tok)
			filt_toks.insert(0,"[CLS]")
			filt_toks.append("[SEP]")

			all_sents.append(filt_toks)

	return all_sents


def get_batches(sentences, max_batch, tokenizer):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				toks=tokenizer.tokenize(word)
				length+=len(toks)

			if length> maxLen:
				maxLen=length

		all_data=[]
		all_masks=[]
		all_labels=[]
		all_transforms=[]

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			labels=[]
			transform=[]

			all_toks=[]
			n=0
			for idx, word in enumerate(sentence):
				toks=tokenizer.tokenize(word)
				all_toks.append(toks)
				n+=len(toks)

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(n))
				for j in range(cur,cur+len(toks)):
					ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				labels.append(1)

			all_data.append(tok_ids)
			all_masks.append(input_mask)
			all_labels.append(labels)
			all_transforms.append(transform)

		lengths = np.array([len(l) for l in all_data])

		# Note sequence must be ordered from shortest to longest so current_batch will work
		ordering = np.argsort(lengths)
		
		ordered_data = [None for i in range(len(all_data))]
		ordered_masks = [None for i in range(len(all_data))]
		ordered_labels = [None for i in range(len(all_data))]
		ordered_transforms = [None for i in range(len(all_data))]
		

		for i, ind in enumerate(ordering):
			ordered_data[i] = all_data[ind]
			ordered_masks[i] = all_masks[ind]
			ordered_labels[i] = all_labels[ind]
			ordered_transforms[i] = all_transforms[ind]

		batched_data=[]
		batched_mask=[]
		batched_labels=[]
		batched_transforms=[]

		i=0
		current_batch=max_batch

		while i < len(ordered_data):

			batch_data=ordered_data[i:i+current_batch]
			batch_mask=ordered_masks[i:i+current_batch]
			batch_labels=ordered_labels[i:i+current_batch]
			batch_transforms=ordered_transforms[i:i+current_batch]

			max_len = max([len(sent) for sent in batch_data])
			max_label = max([len(label) for label in batch_labels])

			for j in range(len(batch_data)):
				
				blen=len(batch_data[j])
				blab=len(batch_labels[j])

				for k in range(blen, max_len):
					batch_data[j].append(0)
					batch_mask[j].append(0)
					for z in range(len(batch_transforms[j])):
						batch_transforms[j][z].append(0)

				for k in range(blab, max_label):
					batch_labels[j].append(-100)

				for k in range(len(batch_transforms[j]), max_label):
					batch_transforms[j].append(np.zeros(max_len))

			batched_data.append(torch.LongTensor(batch_data))
			batched_mask.append(torch.FloatTensor(batch_mask))
			batched_labels.append(torch.LongTensor(batch_labels))
			batched_transforms.append(torch.FloatTensor(batch_transforms))

			bsize=torch.FloatTensor(batch_transforms).shape
			
			i+=current_batch

			# adjust batch size; sentences are ordered from shortest to longest so decrease as they get longer
			if max_len > 100:
				current_batch=12
			if max_len > 200:
				current_batch=6

		return batched_data, batched_mask, batched_transforms, ordering

class BertLatin(nn.Module):

	def __init__(self, bertPath=None):
		super(BertLatin, self).__init__()

		self.bert = BertModel.from_pretrained(bertPath)
		self.bert.eval()
		
	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)
		sequence_outputs, pooled_outputs = self.bert.forward(input_ids, token_type_ids=None, attention_mask=attention_mask)

		all_layers=sequence_outputs
		out=torch.matmul(transforms,all_layers)
		return out

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
	
	parser.add_argument('-f','--file', help='File containing data to get BERT representations for', required=False)
	parser.add_argument('-o','--outfile', help='File to write BERT representations to', required=False)
	
	args = vars(parser.parse_args())

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]			
	encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
	wp_tokenizer = LatinTokenizer(encoder)
	model = BertLatin(bertPath=bertPath)
	model.to(device)

	filename=args["file"]
	outFileName=args["outfile"]


	sents=read_file(filename)
	batch_size=32
	batched_data, batched_mask, batched_transforms, ordering=get_batches(sents, batch_size, wp_tokenizer)

	ordered_preds=[]
	for b in range(len(batched_data)):
		size=batched_transforms[b].shape
		b_size=size[0]
		berts=model.forward(batched_data[b], attention_mask=batched_mask[b], transforms=batched_transforms[b])
		berts=berts.detach()
		berts=berts.cpu()
		for row in range(b_size):
			ordered_preds.append([np.array(r) for r in berts[row]])

	preds_in_order = [None for i in range(len(sents))]
	for i, ind in enumerate(ordering):
		preds_in_order[ind] = ordered_preds[i]


	with open(outFileName, "w", encoding="utf-8") as out:
		for idx, sentence in enumerate(sents):

			out.write("%s\t%s\n" % ("[CLS]", ' '.join("%.5f" % x for x in preds_in_order[idx][0])))

			for t_idx in range(1, len(sentence)-1):
				token=sentence[t_idx]
				
				pred=preds_in_order[idx][t_idx]

				out.write("%s\t%s\n" % (token, ' '.join("%.5f" % x for x in pred)))

			out.write("%s\t%s\n" % ("[SEP]", ' '.join("%.5f" % x for x in preds_in_order[idx][len(sentence)-1])))
			out.write("\n")

