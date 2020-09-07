import sys
from bs4 import BeautifulSoup

def proc(filename):
	file=open(filename)

	page=""
	for line in file.readlines():  
		page+=line.rstrip() + " "

	file.close()
	soup=BeautifulSoup(page, features="lxml")

	entries=soup.findAll("entryfree", {"type":"main"})
	for entry in entries:
		key=entry["key"]
		orth=entry.findAll(["orth"], {"extent":"full"})[0].text

		senses=entry.findAll(["sense"])
		currentLevel1=None
		for sense in senses:
			n=sense["n"]
			level=int(sense["level"])

			if n == "I" or n == "II" or n == "III" or n == "IV" or n == "V":
				currentLevel1=n
			cites=sense.findAll(["cit"])
			for cite in cites:
				quotes=cite.findAll("quote", {"lang":"la"})
				bibl=None
				try:
					bibl=cite.findAll("bibl")[0].text
				except:
					pass
				for quote in quotes:
					text=quote.text
					print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (orth, key, currentLevel1, n, level, text, bibl))


proc(sys.argv[1])

