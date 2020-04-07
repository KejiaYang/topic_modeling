from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from pprint import pprint
from gensim.models import CoherenceModel

import re
import csv
import pandas as pd

import pyLDAvis.gensim
import pickle 
import pyLDAvis



def split_str(str):
	str = str.replace("\\n", " ")
	ret = re.split("\[\'|\'\]|\'\, \'|\"\, \"|\[\"|\"\]", str)
	return ret


def read_content(filename):
	docs = []
	asin_list = []
	test_docs = []
	with open(filename) as tsvfile:
		reader = csv.DictReader(tsvfile, dialect = 'excel-tab')
		for count, row in enumerate(reader):
			asin_list.append(row['asin'])
			temp = ''
			for ele in split_str(row['description']):
				if len(ele) != 0:
					temp = temp + ' ' + remove_stopwords(ele)
					docs.append(remove_stopwords(ele))
			test_docs.append(temp)

	return docs, asin_list, test_docs


# Tokenize the documents.
def tokenize(docs):
	# Split the documents into tokens.
	tokenizer = RegexpTokenizer(r'\w+')
	for idx in range(len(docs)):
		docs[idx] = docs[idx].lower()  # Convert to lowercase.
		docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

	# Remove numbers, but not words that contain numbers.
	docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

	# Remove words that are only one character.
	docs = [[token for token in doc if len(token) > 1] for doc in docs]

	return docs


# Lemmatize the documents.
def lemmatize(docs):
	lemmatizer = WordNetLemmatizer()
	docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

	return docs


# Compute bigrams.
def compute_bigrams(docs):
	# Add bigrams and trigrams to docs (only ones that appear 2 times or more).
	bigram = Phrases(docs, min_count=2)
	for idx in range(len(docs)):
		for token in bigram[docs[idx]]:
			if '_' in token:
				# Token is a bigram, add to document.
				docs[idx].append(token)

	return docs


# Remove rare words and common words based on their document frequency.
def remove_rare_common_words(docs):
	# Create a dictionary representation of the documents.
	dictionary = Dictionary(docs)

	# Filter out words that occur less than 20 documents, or more than 50% of the documents.
	dictionary.filter_extremes(no_below=300, no_above = 1.0)

	return dictionary


# Transform the documents to a vectorized form by computing the frequency of each word, including the bigrams.
def vectorize(dictionary, docs):
	# Bag-of-words representation of the documents.
	corpus = [dictionary.doc2bow(doc) for doc in docs]

	return corpus


# Train LDA model.
def train(dictionary, corpus,num_topics,docs):
	# Set training parameters.
	num_topics = num_topics
	chunksize = 2000
	passes = 1
	iterations = 50
	eval_every = None

	# Make a index to word dictionary.
	temp = dictionary[0]
	id2word = dictionary.id2token

	model = LdaModel(
	    corpus=corpus,
	    id2word=id2word,
	    chunksize=chunksize,
	    alpha='auto',
	    eta='auto',
	    iterations=iterations,
	    num_topics=num_topics,
	    passes=passes,
	    eval_every=eval_every
	)

    # Compute Coherence Score
	coherence_model_lda = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('Coherence: %.4f.' % coherence_lda)

	return (model,id2word)


def main():
	# Training data preprocessing
	docs, asin_list, test_docs = read_content('../data/product_description_complete.tsv')
	docs = tokenize(docs)
	docs = lemmatize(docs)
	docs = compute_bigrams(docs)
	dictionary = remove_rare_common_words(docs)
	corpus = vectorize(dictionary, docs)

	# Train model
	(model,id2word) = train(dictionary,corpus,17,docs)

	# Print topics
	for i in range(17):
		topics = model.show_topic(i)
		print(i,[topic[0] for topic in topics])


	# Testing data preprocessing
	test_docs = tokenize(test_docs)
	test_docs = lemmatize(test_docs)
	test_docs = compute_bigrams(test_docs)
	test_dictionary = remove_rare_common_words(test_docs)
	test_corpus = vectorize(test_dictionary, test_docs)

	# Write predicted results
	with open('../results/product_description_complete.tsv', 'wt') as tsvfile:
    	writer = csv.writer(tsvfile, delimiter='\t')
    	writer.writerow(["asin", "topic_distribution"])
    	for c in test_corpus:
        	writer.writerow([asin_list[i], model[c]])


    # Visualize the topics (the following code can only be run on Notebook)
	pyLDAvis.enable_notebook()
	LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
	LDAvis_prepared




if __name__ == '__main__':
	main()