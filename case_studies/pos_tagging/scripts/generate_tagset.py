import sys

def proc(filenames):
	tags={}
	for filename in filenames:
		with open(filename) as file:
			for line in file:
				if line.startswith("#") or len(line.rstrip()) == 0:
					continue

				cols=line.rstrip().split("\t")

				label=cols[3]
				tags[label]=1

	for idx, tag in enumerate(tags):
		print("%s\t%s" % (tag, idx))


proc(sys.argv[1:])