import os,sys,argparse
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
import sequence_eval, sequence_reader
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

random.seed(1)
torch.manual_seed(0)
np.random.seed(0)

batch_size=32
dropout_rate=0.25
HIDDEN_DIM=100
VOCAB=1000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')

def read_annotations(filename, tagset, labeled):

	""" Read tsv data and return sentences and [word, tag, sentenceID, filename] list """

	with open(filename, encoding="utf-8") as f:
		sentence = []
		sentences = []
		sentenceID=0
		for line in f:
			if line.startswith("#"):
				continue

			if len(line) > 0:
				if line == '\n':
					sentenceID+=1

					sentences.append(sentence)
					sentence = []


				else:
					data=[]
					split_line = line.rstrip().split('\t')

					# LOWERCASE for latin

					word=split_line[1].lower()
					label=0
					if labeled:
						label=split_line[3]
					
					data.append(word)
					data.append(tagset[label])
					
					data.append(sentenceID)
					data.append(filename)

					sentence.append(data)
		
		if len(sentence) > 2:
			sentences.append(sentence)

	return sentences


def prepare_annotations_from_file(filename, tagset, labeled=True):

	""" Read a single file of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	annotations = read_annotations(filename, tagset, labeled)
	sentences += annotations
	return sentences



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
			val = np.array(cols[1:], dtype="float")
			word = cols[0]
			embeddings[idx + 2] = val
			vocab[word] = idx + 2

	vocab["[MASK]"]=0
	vocab["[UNK]"]=1

	return torch.FloatTensor(embeddings), vocab


class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, pretrained_embeddings, vocab, tagset_size):
		super(LSTMTagger, self).__init__()

		vocab_size=len(vocab)
		self.num_labels=tagset_size
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=4)

		# The linear layer that maps from hidden state space to tag space
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


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='{train,test,predict,predictBatch}', required=True)
	parser.add_argument('-f','--modelFile', help='File to write model to/read from', required=False)
	parser.add_argument('--max_epochs', help='max number of epochs', required=False)
	parser.add_argument('-e','--embeddingsFile', help='word2vec embeddings', required=False)
	parser.add_argument('-r','--trainFile', help='File containing training data', required=False)
	parser.add_argument('-z','--testFile', help='File containing test data', required=False)
	parser.add_argument('-d','--devFile', help='File containing dev data', required=False)
	parser.add_argument('-g','--tagFile', help='File listing tags + tag IDs', required=False)


	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]

	tagFile=args["tagFile"]
	tagset=sequence_reader.read_tagset(tagFile)

	embeddingsFile=args["embeddingsFile"]

	modelFile=args["modelFile"]
	max_epochs=args["max_epochs"]

	embeddings, vocab=read_embeddings(embeddingsFile, vocab_size=VOCAB)
	_, EMBEDDING_DIM=embeddings.shape
	

	epochs=100

	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, embeddings, vocab, len(tagset))
	model.to(device)

	if mode == "train":

		train_file=args["trainFile"]
		dev_file=args["devFile"]
	
		metric=sequence_eval.get_accuracy

		trainSentences = prepare_annotations_from_file(train_file, tagset)
		batch_sents, batch_labels, batch_lengths=model.get_batches(trainSentences, batch_size, vocab)

		if dev_file is not None:
			devSentences = prepare_annotations_from_file(dev_file, tagset)
			batch_sents_dev, batch_labels_dev, batch_lengths_dev=model.get_batches(devSentences, batch_size, vocab)

		learning_rate=0.001
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		maxScore=0
		best_idx=0
		patience=10

		epochs=100
		if max_epochs is not None:
			epochs=int(max_epochs)

		for epoch in range(epochs):
			model.train()
			big_loss=0
			for b in range(len(batch_sents)):
				
				loss = model(indices=batch_sents[b], lengths=batch_lengths[b], labels=batch_labels[b])
				big_loss+=loss
				loss.backward()
				optimizer.step()
				model.zero_grad()

			print("loss: ", big_loss)
			sys.stdout.flush()

			score=0.

			if dev_file is not None:
				print("\n***EVAL***\n")

				c, t=model.evaluate(batch_sents_dev, batch_lengths_dev, batch_labels_dev, metric, tagset)
				score=float(c)/t
				print("Dev accuracy: %.3f %s" % (score, t))

			sys.stdout.flush()
			if dev_file is None or score > maxScore:
				torch.save(model.state_dict(), modelFile)
				maxScore=score
				best_idx=epoch

			if epoch-best_idx > patience:
				print ("Stopping training at epoch %s" % epoch)
				break


	elif mode == "test":

		metric=metric=sequence_eval.get_accuracy

		test_file=args["testFile"]

		testSentences = prepare_annotations_from_file(test_file, tagset)
		batch_sents_test, batch_labels_test, batch_lengths_test=model.get_batches(testSentences, batch_size, vocab)

		model.load_state_dict(torch.load(modelFile, map_location=device))
		
		c, t=model.evaluate(batch_sents_test, batch_lengths_test, batch_labels_test, metric, tagset)
		score=float(c)/t
		print("Test accuracy: %.3f\t%s" % (score, t))


