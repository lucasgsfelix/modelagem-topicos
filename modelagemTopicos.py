# coding: utf-8
# user/bin/python
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import re
from gensim import corpora

##### esse código será responsável por fazer a modelagem de tópicos no nosso trabalho

def leitura(arquivo, posicao):

	arq = open(arquivo, "r")
	info = arq.read()
	info = unicode(info, 'utf-8')
	arq.close()
	info = info.split('\n')
	data = []

	for i in range(0, len(info)):

		info[i] = info[i].split('\t')
		data.append(info[i][posicao]) ### posicao é 7 para base do reclame aqui
									### tem que olhar qual posição na base do twitter

	return data

def preProcessamento(info): ### pré processando os dados
	
	stop_free = " ".join([i for i in info.lower().split() if i not in stop]) ## retirando stop words
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude) ### retirando pontuação

	return punc_free


def removeWords(adjectives, text):
	
	i=0
	while(i<len(text)):

		j=0
		while(j<len(text[i])):

			if len([(a.end()) for a in list(re.finditer(text[i][j], adjectives))]) != 0:

				text[i].pop(j)
				j=j-1

			j=j+1

			if len(text[i]) == 0:
				break

		if len(text) == 0:
			break

		i=i+1

	return text

def readWords(file_name):

	text_file = open(file_name, 'r')
	info = text_file.read()
	
	text_file.close()

	return info

if __name__ == '__main__':
	
	info = leitura("reclameAqui.txt", 6)
	stop = set(stopwords.words('portuguese'))
	exclude = set(string.punctuation) 

	for i in range(0, len(info)):
		### o que eu preciso retirar <br
		info[i] = info[i].replace('<br', ' ')

	info = [preProcessamento(data).split() for data in info] ## ao final tenho tudo pre processado
	adjectives = readWords('lista_adjetivos')
	stop = readWords('stop_words')
	info = removeWords(adjectives, info) ### retirando adjetivos
	info = removeWords(stop, info) ### retirando stop words


	#### parte de modelagem de tópicos
	dictionary = corpora.Dictionary(info)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in info]
	Lda = gensim.models.ldamodel.LdaModel
	ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

	print(ldamodel.print_topics(num_topics=10, num_words=3))
		













