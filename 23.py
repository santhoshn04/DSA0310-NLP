import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = """
Coherence in writing means that the ideas in each sentence and paragraph are connected and logical.
It helps the reader to follow your arguments and understand your point. Without coherence, the text may be confusing.
There are several ways to achieve coherence in your writing, such as using transition words and repeating key concepts.
"""
sentences = sent_tokenize(text)
stop_words = set(stopwords.words("english"))
word_tokens = [word_tokenize(sentence) for sentence in sentences]
filtered_tokens = [[word for word in words if word.lower() not in stop_words] for words in word_tokens]
preprocessed_sentences = [" ".join(words).lower() for words in filtered_tokens]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
coherence_score = cosine_matrix.sum(axis=1).mean()
print("Coherence Score:", coherence_score)
