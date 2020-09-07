import argparse, sys, re
import numpy as np
from numpy import linalg as LA
from gen_berts import LatinBERT
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
from tqdm import tqdm


PINK = '\033[95m'
ENDC = '\033[0m'

def proc(filenames):
	matrix_all=[]
	sents_all=[]
	sent_ids_all=[]
	toks_all=[]
	position_in_sent_all=[]
	doc_ids=[]

	num_parallel_processes = 10
	vals=Parallel(n_jobs=num_parallel_processes)(
			delayed(proc_doc)(f) for f in tqdm(filenames))

	for matrix, sents, sent_ids, toks, position_in_sent, filename in vals:
		matrix_all.append(matrix)
		sents_all.append(sents)
		sent_ids_all.append(sent_ids)
		toks_all.append(toks)
		position_in_sent_all.append(position_in_sent)
		doc_ids.append(filename)

	return matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids

def proc_doc(filename):
	berts=[]
	toks=[]
	sent_ids=[]
	sentid=0
	position_in_sent=[]
	p=0
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			if len(cols) == 2:
				word=cols[0]
				bert=np.array([float(x) for x in cols[1].split(" ")])
				bert=bert/LA.norm(bert)
				toks.append(word)
				berts.append(bert)
				sent_ids.append(sentid)
				position_in_sent.append(p)
				p+=1
			else:
				sentid+=1
				p=0

	sents=[]
	lastid=0
	current_sent=[]
	for s, t in zip(sent_ids, toks):
		if s != lastid:
			sents.append(current_sent)
			current_sent=[]
		lastid=s
		current_sent.append(t)

	matrix=np.asarray(berts)
	
	return matrix, sents, sent_ids, toks, position_in_sent, filename


def get_window(pos, sentence, window):
	start=pos - window if pos - window >= 0 else 0
	end=pos + window + 1 if pos + window + 1 < len(sentence) else len(sentence)
	return "%s %s%s%s %s" % (' '.join(sentence[start:pos]), PINK, sentence[pos], ENDC, ' '.join(sentence[pos+1:end]))

def compare_one(idx, matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids, target_bert):
	c_matrix=matrix_all[idx]
	c_sents=sents_all[idx]
	c_sent_ids=sent_ids_all[idx]
	c_toks=toks_all[idx]
	c_pos=position_in_sent_all[idx]
	doc_id=doc_ids[idx]
	similarity=np.dot(c_matrix,target_bert)
	argsort=np.argsort(-similarity)
	len_s,=similarity.shape
	vals5=[]
	vals10=[]
	for i in range(min(100,len_s)):
		tid=argsort[i]
		if tid < len(c_sent_ids) and tid < len(c_pos) and c_sent_ids[tid] < len(c_sents):
			wind10=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 10)
			wind5=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 5)
			vals5.append((similarity[tid], wind5, doc_id ))
			vals10.append((similarity[tid], wind10, doc_id ))

	return vals5, vals10

def compare(berts, target_bert, outputDir, query, sent):

	vals=[]
	outfile="%s/%s_%s" % (outputDir, query, re.sub(" ", "_", sent))
	out=open(outfile, "w", encoding="utf-8")

	matrix_all, sents_all, sent_ids_all, toks_all, position_in_sent_all, doc_ids=berts

	for idx in range(len(doc_ids)):
		c_matrix=matrix_all[idx]
		c_sents=sents_all[idx]
		c_sent_ids=sent_ids_all[idx]
		c_toks=toks_all[idx]
		c_pos=position_in_sent_all[idx]
		doc_id=doc_ids[idx]

		similarity=np.dot(c_matrix,target_bert)
		argsort=np.argsort(-similarity)
		len_s,=similarity.shape
		for i in range(min(100,len_s)):
			tid=argsort[i]
			if tid < len(c_sent_ids) and tid < len(c_pos) and c_sent_ids[tid] < len(c_sents):
				wind10=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 10)
				wind5=get_window(c_pos[tid], c_sents[c_sent_ids[tid]], 5)
				out.write("%s\t%s\t%s\n" % (similarity[tid], wind10, doc_id))
				vals.append((similarity[tid], wind5, doc_id ))

	vals=sorted(vals, key=lambda x: x[0])
	for a, b, doc in vals[-25:]:
		print("%.3f\t%s\t%s" % (a, b, doc))

	out.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
	parser.add_argument('-i', '--inputDir', help='input files to search', required=True)
	parser.add_argument('-o', '--outputDir', help='output directory to write results to', required=True)
	
	args = vars(parser.parse_args())

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]			
	inputDir=args["inputDir"]			
	outputDir=args["outputDir"]			

	onlyfiles = [f for f in listdir(inputDir) if isfile(join(inputDir, f))]
	target_files=[]
	for filename in onlyfiles:

		target_files.append("%s/%s" % (inputDir, filename))

	bert=LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)

	berts=proc(target_files)

	print ("> ",)
	line = sys.stdin.readline()
	while line:
		word=line.rstrip()
		toks=line.rstrip().split(" ")
		target_word=toks[0]
		sents=[' '.join(toks[1:])]

		bert_sents=bert.get_berts(sents)[0]
		toks=[]
		target_bert=None
		seen=False
		for idx, (tok, b) in enumerate(bert_sents):
			toks.append(tok)
			if tok == target_word:
				if seen:
					print("more than one instance of %s" % target_word)
					sys.exit(1)
				else:
					target_bert=b
					seen=True
		print(' '.join(toks))
		print("target: %s" % target_word)

		if target_bert is not None:

			target_bert=target_bert/LA.norm(target_bert)
			compare(berts, target_bert, outputDir, target_word, sents[0])

		print ("> ",)
		line = sys.stdin.readline()

