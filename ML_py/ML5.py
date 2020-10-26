from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
print(vectorizer.fit_transform(text))
print(vectorizer.vocabulary_)


# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())