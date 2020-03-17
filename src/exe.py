from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from pprint import pprint

import re
import csv
import pandas as pd

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
					# docs.append(ele)
					docs.append(remove_stopwords(ele))
			# if count == 5:
			# 	break
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
	# dictionary.filter_extremes(no_below=20, no_above=0.5)
	dictionary.filter_extremes(no_below=300, no_above = 1.0)

	return dictionary


# Transform the documents to a vectorized form by computing the frequency of each word, including the bigrams.
def vectorize(dictionary, docs):
	# Bag-of-words representation of the documents.
	corpus = [dictionary.doc2bow(doc) for doc in docs]

	return corpus


# Train LDA model.
def train(dictionary, corpus):
	# Set training parameters.
	num_topics = 20
	chunksize = 2000
	# passes = 20
	passes = 1
	iterations = 50
	eval_every = None  # Don't evaluate model perplexity, takes too much time.

	# Make a index to word dictionary.
	temp = dictionary[0]  # This is only to "load" the dictionary.
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

	top_topics = model.top_topics(corpus) #, num_words=20)

	# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
	avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
	print('Average topic coherence: %.4f.' % avg_topic_coherence)
	pprint(top_topics)

	return model


def format_topics_sentences(ldamodel, corpus, texts):
	# Init output
	# sent_topics_df = pd.DataFrame()

	# # Get main topic in each document
	# for i, row_list in enumerate(ldamodel[corpus]):
	# 	row = row_list[0] if ldamodel.per_word_topics else row_list            
	# 	# print(row)
	# 	row = sorted(row, key=lambda x: (x[1]), reverse=True)
	# 	# Get the Dominant topic, Perc Contribution and Keywords for each document
	# 	for j, (topic_num, prop_topic) in enumerate(row):
	# 		if j < 20:  # => dominant topic
	# 			wp = ldamodel.show_topic(topic_num)
	# 			topic_keywords = ", ".join([word for word, prop in wp])
	# 			sent_topics_df = sent_topics_df.append(pd.Series([wp, round(prop_topic,4), topic_keywords, i]), ignore_index=True)
	# 		else:
	# 			break
	# sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'i']

	# # Add original text to the end of the output
	# contents = pd.Series(texts)
	# sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
	# return(sent_topics_df)

	print (len(corpus))




def main():
	docs, asin_list, test_docs = read_content('../data/product_description_truncated.tsv')
	# print(len(docs))
	# print(type(docs[0]))
	docs = tokenize(docs)
	docs = lemmatize(docs)
	docs = compute_bigrams(docs)

	dictionary = remove_rare_common_words(docs)

	corpus = vectorize(dictionary, docs)
	# print('Number of unique tokens: %d' % len(dictionary))
	# print('Number of documents: %d' % len(corpus))
	# for idx, doc in enumerate(dictionary):
	# 	print("Document '{}' key phrases:".format(dictionary[idx]))
	# print(dictionary)
	# print(corpus)

	model = train(dictionary, corpus)
	# # for i in range(20):
	# # 	print (model.show_topic(i))
	# df_topic_sents_keywords = format_topics_sentences(ldamodel=model, corpus=corpus, texts=docs)
	# # # Format
	# # df_dominant_topic = df_topic_sents_keywords.reset_index()
	# # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'i', 'Text']
	# # pd.options.display.max_columns = 50
	# # print(df_dominant_topic.head(50))


	test_docs = tokenize(test_docs)
	test_docs = lemmatize(test_docs)
	test_docs = compute_bigrams(test_docs)

	test_dictionary = remove_rare_common_words(test_docs)

	test_corpus = vectorize(test_dictionary, test_docs)
	i = 0
	for c in test_corpus:
		print (asin_list[i])
		print (model[c])
		if i == 5:
			break
		i = i + 1
	# print (len(test_corpus))



if __name__ == '__main__':
	main()