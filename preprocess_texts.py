# Preprocess poems and concatenate all collections to a single file
import re

# Load and preprocess texts
# See regex expressions, e.g., here: https://www.rexegg.com/regex-quickstart.html
def load_text(path):

	text = open(path, "r").read()

	text = text.replace("ArribaAbajo","")	# Remove spurious words (from Cervantes Virtual)
	text = text.replace("Abajo   ","")	# Remove spurious words (from Cervantes Virtual)
	text = text.replace("Arriba   ","")	# Remove spurious words (from Cervantes Virtual)

	text = re.sub("\[\d+\]","",text)	# Remove notes (from Project Gutenberg)
	text = re.sub("\^\{.*\}","",text)	# Remove characters "^" (from Project Gutenberg)
	text = re.sub("\w\^\w","",text)	# Remove characters "^" (from Project Gutenberg)

	text = text.lower()						# Set to lowercase
	text = re.sub(" \d+", " ", text)		# Remove digits
	text = re.sub("\d+", "", text)		# Remove digits
	text = text.replace("\t","")			# Remove tabs
	text = re.sub("- [a-zA-Z]* -","", text)	# Remove numeration of poems
	text = re.sub("\n\n+", "\n\n", text)		# Remove multiple
	text = re.sub("\s\s^\n+", "", text)		# Remove spaces
	text = text.replace("  ","")			# Remove spaces
	text = text.replace("\n ","\n")			# Remove spaces
	text = text.replace("_","")			# Remove underscores

	return text

# Compile all texts by an author and preprocess them
def preprocess(author):

	# Texts
	filepath = "data/"+author+"/"
	if author=="Lope":
		filenames = ["Rimas","Otros_Sonetos","Rimas_Humanas_y_divinas","Rimas_Sacras","Sonetos_en_comedias_autos_y_entremeses","Sonetos_en_libros"]
	elif author=="Verdaguer":
		filenames = ["L'Atlantida", "Canigo", "Cansons_de_Montserrat", "Oda_a_Barcelona"]
	elif author=="Donne":
		filenames = ["Poems_Donne_1", "No_man_is_an_island"]
	else:
		print("Author not included.")

	# Load texts
	texts = []
	for filename in filenames:
		text = load_text(filepath + filename + ".txt")
		texts.append(text)

	totaltext = "".join(texts)
	#print([totaltext[0:200]])
	newfile = open(filepath+"processed.txt","w")
	newfile.write(totaltext)
	newfile.close()

# Choose author
author = "Lope"
preprocess(author)

author = "Verdaguer"
preprocess(author)

author = "Donne"
preprocess(author)
