import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
OPEN_FILE_NAME = 'news_test'
OPEN_FILE_NAME = r'data/' + OPEN_FILE_NAME + '.json'
OPEN_FILE_DATA = 'news_train_1'
OPEN_FILE_DATA = r'data/' + OPEN_FILE_DATA + '.json'
OPEN_FILE_TAGS = 'tags'
OPEN_FILE_TAGS = r'data/' + OPEN_FILE_TAGS + '.json'
SAVE_FILE_NAME = 'result'
SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.txt'


cats = ['media', 'culture', 'sport', 'business', 'science', 'life', 'style', 'economics', 'forces', 'travel']


def main():
    with open(OPEN_FILE_NAME, encoding='utf-8') as file:
        test_data = json.loads(file.read())
    file.close()
    with open(OPEN_FILE_DATA, encoding='utf-8') as file:
        data = json.loads(file.read())
    file.close()
    with open(OPEN_FILE_TAGS, encoding='utf-8') as file:
        tags = json.loads(file.read())
    file.close()
    return test_data, data, tags


pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', MultinomialNB(alpha=0.00001)),
])


if __name__ == '__main__':
    test_data, data, tags = main()
    _ = pipeline.fit(data, tags)
    predicted = pipeline.predict(test_data)
    result = []
    for i in range(0, len(predicted)):
        result.append(cats[predicted[i]-1])
    res = '\n'.join(result)
    with open(SAVE_FILE_NAME, 'w', encoding='utf-8') as file:
        file.write(res)
    file.close()
