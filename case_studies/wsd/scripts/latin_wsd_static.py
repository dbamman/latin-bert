import os,sys,argparse
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
import sequence_eval
from tensor2tensor.data_generators import text_encoder
import random
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

random.seed(1)
torch.manual_seed(0)
np.random.seed(0)

batch_size=32
dropout_rate=0.25
bert_dim=768
HIDDEN_DIM=100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')

def read_embeddings(filename, vocab_size=10000):

	with open(filename, encoding="utf-8") as file:
		word_embedding_dim = int(file.readline().split(" ")[1])

	vocab = {}

	print(vocab_size, word_embedding_dim)
	embeddings = np.zeros((vocab_size, word_embedding_dim))

	with open(filename, encoding="utf-8") as file:
		for idx, line in enumerate(file):

			if idx + 2 >= vocab_size:
				break

			cols = line.rstrip().split(" ")
			val = np.array(cols[1:])
			word = cols[0]
			embeddings[idx + 2] = val
			vocab[word] = idx + 2

	vocab["[MASK]"]=0
	vocab["[UNK]"]=1

	return torch.FloatTensor(embeddings), vocab

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

class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, pretrained_embeddings, vocab, tagset_size):
		super(LSTMTagger, self).__init__()

		vocab_size=len(vocab)
		self.num_labels=tagset_size
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
		self.dropout = nn.Dropout(p=dropout_rate)

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

		self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

	def forward(self, indices=None, lengths=None, labels=None):
		indices = torch.LongTensor(indices).to(device)
		lengths = torch.LongTensor(lengths).to(device)

		if labels is not None:
			labels = torch.LongTensor(labels).to(device)

		embeddings = self.word_embeddings(indices)

		embeddings=self.dropout(embeddings)

		packed_input_embs = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
		packed_output, _ = self.lstm(packed_input_embs)
		padded_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

		padded_output=self.dropout(padded_output)

		logits = self.hidden2tag(padded_output)

		if labels is not None:

			loss_fct = CrossEntropyLoss(ignore_index=-100)
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits


	def evaluate(self, dev_indices, dev_lengths, dev_labels, metric, tagset):
		num_labels=len(tagset)

		model.eval()

		with torch.no_grad():

			ordered_preds=[]

			all_preds=[]
			all_golds=[]

			for b in range(len(dev_lengths)):

				logits = self.forward(indices=dev_indices[b], lengths=dev_lengths[b])

				logits=logits.cpu()

				ordered_preds += [np.array(r) for r in logits]
				size=dev_labels[b].shape

				logits=logits.view(-1, size[1], num_labels)

				for row in range(size[0]):
					for col in range(size[1]):
						if dev_labels[b][row][col] != -100:
							pred=np.argmax(logits[row][col])
							all_preds.append(pred.cpu().numpy())
							all_golds.append(dev_labels[b][row][col].cpu().numpy())
			
			cor=0.
			tot=0.
			for i in range(len(all_golds)):
				if all_golds[i] == all_preds[i]:
					cor+=1
				tot+=1

	

			return cor, tot



	def get_batches(self, sentences, max_batch, vocab):

		random.shuffle(sentences)

		all_sents=[]
		all_labels=[]
		for sentence in sentences:

			toks=[ word[0] for word in sentence ]
			toks=[]
			for word in sentence:
				tok=word[0].lower()
				if tok in vocab:
					toks.append(vocab[tok])
				else:
					toks.append(vocab["[UNK]"])

			labs=[ word[1] for word in sentence ]
			
			all_sents.append(toks)
			all_labels.append(labs)

		lengths = np.array([len(l) for l in all_sents])

		batch_sents=[]
		batch_labels=[]
		batch_lengths=[]

		for i in range(0, len(lengths), max_batch):
			b_sents=[]
			b_labs=[]

			this_lengths=lengths[i:i+max_batch]
			max_length=max(this_lengths)
			for j in range(len(all_sents[i:i+max_batch])):
				sent=all_sents[i+j]
				labs=all_labels[i+j]
				for k in range(len(sent), max_length):
					sent.append(0)
					labs.append(-100)
				b_sents.append(sent)
				b_labs.append(labs)

			b_sents=torch.LongTensor(b_sents)
			b_labs=torch.LongTensor(b_labs)

			batch_sents.append(b_sents)
			batch_labels.append(b_labs)
			batch_lengths.append(torch.LongTensor(this_lengths))
		
		return batch_sents, batch_labels, batch_lengths
	

def get_splits(data):
	trains=[]
	tests=[]
	devs=[]

	for i in range(10):
		trains.append([])
		tests.append([])
		devs.append([])

	for idx, sent in enumerate(data[0]):
		testFold=idx % 10
		devFold=testFold-1
		if devFold == -1:
			devFold=9

		for i in range(10):
			if i == testFold:
				tests[i].append(sent)
			elif i == devFold:
				devs[i].append(sent)
			else:
				trains[i].append(sent)
	
	for idx, sent in enumerate(data[1]):
		testFold=idx % 10
		devFold=testFold-1
		if devFold == -1:
			devFold=9

		for i in range(10):
			if i == testFold:
				tests[i].append(sent)
			elif i == devFold:
				devs[i].append(sent)
			else:
				trains[i].append(sent)

	for i in range(10):
		random.shuffle(trains[i])
		random.shuffle(tests[i])
		random.shuffle(devs[i])

	return trains, devs, tests

def get_labs(before, target, after, label):
	sent=[]
	for word in before.split(" "):
		sent.append((word, -100))
	sent.append((target, label))
	for word in after.split(" "):
		sent.append((word, -100))
	return sent

def read_data(filename):
	lemmas={}
	with open(filename) as file:
		for line in file:
			cols=line.split("\t")
			lemma=cols[0]
			label=cols[1]
			before=cols[2]
			target=cols[3]
			after=cols[4].rstrip()
			if lemma not in lemmas:
				lemmas[lemma]={}
				lemmas[lemma][0]=[]
				lemmas[lemma][1]=[]
				
			if label == "I":
				lemmas[lemma][0].append(get_labs(before, target, after, 0))
			elif label == "II":
				lemmas[lemma][1].append(get_labs(before, target, after, 1))

	return lemmas

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='{train,test,predict,predictBatch}', required=True)
	parser.add_argument('-f','--modelFile', help='File to write model to/read from', required=False)
	parser.add_argument('--max_epochs', help='max number of epochs', required=False)
	parser.add_argument('-i','--inputFile', help='WSD data', required=False)
	parser.add_argument('-e','--embeddingsFile', help='word2vec embeddings', required=False)


	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]
	
	inputFile=args["inputFile"]
	embeddingsFile=args["embeddingsFile"]

	modelFile=args["modelFile"]
	max_epochs=args["max_epochs"]

	data=read_data(inputFile)

	embeddings, vocab=read_embeddings(embeddingsFile, vocab_size=100000)
	_, EMBEDDING_DIM=embeddings.shape

	tagset={0:0, 1:1}

	epochs=100

	devCors=[0.]*epochs
	testCors=[0.]*epochs
	devN=[0.]*epochs
	testN=[0.]*epochs
	
	if mode == "train":
	
		metric=sequence_eval.get_accuracy

		for lemma in data:

			cor=0.
			tot=0.

			print(lemma)
			trains, devs, tests=get_splits(data[lemma])

			trainData=trains[0]
			testData=tests[0]
			devData=devs[0]
			print("train: %s, dev: %s, test: %s" % (len(trainData), len(devData), len(testData)))

			model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, embeddings, vocab, 2)

			model.to(device)

			batch_sents, batch_labels, batch_lengths=model.get_batches(trainData, batch_size, vocab)
			batch_sents_dev, batch_labels_dev, batch_lengths_dev=model.get_batches(devData, batch_size, vocab)
			
			batch_sents_test, batch_labels_test, batch_lengths_test=model.get_batches(testData, batch_size, vocab)

			learning_rate_fine_tuning=0.001
			optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tuning)
			
			maxScore=0
			best_idx=0
			patience=10

			if max_epochs is not None:
				epochs=int(max_epochs)

			for epoch in range(epochs):
				model.train()
				big_loss=0
				for b in range(len(batch_lengths)):
					if b % 10 == 0:
						# print(b)
						sys.stdout.flush()
					
					loss = model(indices=batch_sents[b], lengths=batch_lengths[b], labels=batch_labels[b])
					big_loss+=loss
					loss.backward()
					optimizer.step()
					model.zero_grad()

				c, t=model.evaluate(batch_sents_dev, batch_lengths_dev, batch_labels_dev, metric, tagset)
				devCors[epoch]+=c
				devN[epoch]+=t

				c, t=model.evaluate(batch_sents_test, batch_lengths_test, batch_labels_test, metric, tagset)
				testCors[epoch]+=c
				testN[epoch]+=t

			for epoch in range(epochs):
				devAcc=devCors[epoch]/devN[epoch]
				print("DEV:\t%s\t%.3f\t%s\t%s" % (epoch, devAcc, lemma, devN[epoch]))

		maxAcc=0
		bestDevEpoch=None
		for i in range(epochs):
			acc=devCors[i]/devN[i]
			if acc > maxAcc:
				maxAcc=acc
				bestDevEpoch=i

		testAcc=testCors[bestDevEpoch]/testN[bestDevEpoch]

		print("OVERALL:\t%s\t%.3f\t%s" % (bestDevEpoch, testAcc, testN[bestDevEpoch]))

