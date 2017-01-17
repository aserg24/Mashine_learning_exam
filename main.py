import json
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

OPEN_FILE_DATA = 'news_train_1'
#SAVE_FILE_NAME = 'test'
OPEN_FILE_DATA = r'data/' + OPEN_FILE_DATA + '.json'
#SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.json'
OPEN_FILE_TAGS = 'tags'
OPEN_FILE_TAGS = r'data/' + OPEN_FILE_TAGS + '.json'

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),

}


def main():
    with open(OPEN_FILE_DATA, encoding='utf-8') as file:
        data = json.loads(file.read())
    file.close()
    with open(OPEN_FILE_TAGS, encoding='utf-8') as file:
        tags = json.loads(file.read())
    file.close()
    return data, tags

if __name__ == '__main__':
    data, tags = main()

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print('Start')
    print('pipeline:', [name for name, _ in pipeline.steps])
    print('parameters:')
    print(parameters)
    t0 = time()
    grid_search.fit(data, tags)
    print('done in {}'.format(time() - t0))
    print()

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('{}: {}'.format(param_name, best_parameters[param_name]))