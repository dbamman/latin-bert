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

random.seed(1)
torch.manual_seed(0)
np.random.seed(0)

batch_size=32
dropout_rate=0.25
bert_dim=768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')


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


class BertForSequenceLabeling(nn.Module):

	def __init__(self, tokenizerPath=None, bertPath=None, freeze_bert=False, num_labels=2):
		super(BertForSequenceLabeling, self).__init__()

		encoder = text_encoder.SubwordTextEncoder(tokenizerPath)

		self.tokenizer = LatinTokenizer(encoder)
		self.num_labels = num_labels
		self.bert = BertModel.from_pretrained(bertPath)

		self.bert.eval()
		
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

		self.dropout = nn.Dropout(dropout_rate)
		self.classifier = nn.Linear(bert_dim, num_labels)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None, labels=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)

		if labels is not None:
			labels = labels.to(device)

		sequence_outputs, pooled_outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
		all_layers=sequence_outputs
		out=torch.matmul(transforms,all_layers)

		logits = self.classifier(out)

		if labels is not None:

			loss_fct = CrossEntropyLoss(ignore_index=-100)
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits

	def predict(self, dev_file, tagset, outfile):

		num_labels=len(tagset)

		rev_tagset={tagset[v]:v for v in tagset}

		dev_orig_sentences = sequence_reader.prepare_annotations_from_file(dev_file, tagset, labeled=False)
		dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(dev_orig_sentences, batch_size)

		model.eval()

		bcount=0

		with torch.no_grad():

			ordered_preds=[]

			all_preds=[]
			all_golds=[]

			for b in range(len(dev_batched_data)):
				size=dev_batched_transforms[b].shape
				
				b_size=size[0]
				b_size_labels=size[1]
				b_size_orig=size[2]

				logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])
				
				logits=logits.view(-1, b_size_labels, num_labels)

				logits=logits.cpu()

				preds=np.argmax(logits, axis=2)

				for row in range(b_size):
					ordered_preds.append([np.array(r) for r in preds[row]])
	
			preds_in_order = [None for i in range(len(dev_orig_sentences))]
			for i, ind in enumerate(dev_ordering):
				preds_in_order[ind] = ordered_preds[i]
			
			with open(outfile, "w", encoding="utf-8") as out:
				for idx, sentence in enumerate(dev_orig_sentences):

					# skip [CLS] and [SEP] tokens
					for t_idx in range(1, len(sentence)-1):
						sent_list=sentence[t_idx]
						token=sent_list[0]
						s_idx=sent_list[2]
						filename=sent_list[3]

						pred=preds_in_order[idx][t_idx]

						out.write("%s\t%s\n" % (token, rev_tagset[int(pred)]))

					# longer than just "[CLS] [SEP]"
					if len(sentence) > 2:
						out.write("\n")

	def evaluate(self, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, tagset):
		num_labels=len(tagset)

		model.eval()

		with torch.no_grad():

			ordered_preds=[]

			all_preds=[]
			all_golds=[]

			for b in range(len(dev_batched_data)):

				logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])

				logits=logits.cpu()

				ordered_preds += [np.array(r) for r in logits]
				size=dev_batched_labels[b].shape

				logits=logits.view(-1, size[1], num_labels)

				for row in range(size[0]):
					for col in range(size[1]):
						if dev_batched_labels[b][row][col] != -100:
							pred=np.argmax(logits[row][col])
							all_preds.append(pred.cpu().numpy())
							all_golds.append(dev_batched_labels[b][row][col].cpu().numpy())
			
			cor=0.
			tot=0.
			for i in range(len(all_golds)):
				if all_golds[i] == all_preds[i]:
					cor+=1
				tot+=1

	

			return cor, tot



	def get_batches(self, sentences, max_batch):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				toks=self.tokenizer.tokenize(word[0])
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
				toks=self.tokenizer.tokenize(word[0])
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

				tok_ids.extend(self.tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				labels.append(int(word[1]))

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

		return batched_data, batched_mask, batched_labels, batched_transforms, ordering


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
	parser.add_argument('--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)	
	parser.add_argument('-f','--modelFile', help='File to write model to/read from', required=False)
	parser.add_argument('--max_epochs', help='max number of epochs', required=False)
	parser.add_argument('-i','--inputFile', help='WSD data', required=False)


	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]
	
	inputFile=args["inputFile"]

	modelFile=args["modelFile"]
	max_epochs=args["max_epochs"]

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]

	data=read_data(inputFile)


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

			model = BertForSequenceLabeling(tokenizerPath=tokenizerPath, bertPath=bertPath, freeze_bert=False, num_labels=2)

			model.to(device)

			batched_data, batched_mask, batched_labels, batched_transforms, ordering=model.get_batches(trainData, batch_size)
			dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(devData, batch_size)
			test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, test_ordering=model.get_batches(testData, batch_size)

			learning_rate_fine_tuning=0.00005
			optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tuning)
			
			maxScore=0
			best_idx=0

			if max_epochs is not None:
				epochs=int(max_epochs)

			for epoch in range(epochs):
				model.train()
				big_loss=0
				for b in range(len(batched_data)):
					if b % 10 == 0:
						# print(b)
						sys.stdout.flush()
					
					loss = model(batched_data[b], token_type_ids=None, attention_mask=batched_mask[b], transforms=batched_transforms[b], labels=batched_labels[b])
					big_loss+=loss
					loss.backward()
					optimizer.step()
					model.zero_grad()

				c, t=model.evaluate(dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, tagset)
				devCors[epoch]+=c
				devN[epoch]+=t

				c, t=model.evaluate(test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, metric, tagset)
				testCors[epoch]+=c
				testN[epoch]+=t

			for epoch in range(epochs):
				devAcc=devCors[epoch]/devN[epoch]
				print("DEV:\t%s\t%.3f\t%s\t%s" % (epoch, devAcc, lemma, devN[epoch]))
				sys.stdout.flush()

		maxAcc=0
		bestDevEpoch=None
		for i in range(epochs):
			acc=devCors[i]/devN[i]
			if acc > maxAcc:
				maxAcc=acc
				bestDevEpoch=i

		testAcc=testCors[bestDevEpoch]/testN[bestDevEpoch]

		print("OVERALL:\t%s\t%.3f\t%s" % (bestDevEpoch, testAcc, testN[bestDevEpoch]))


