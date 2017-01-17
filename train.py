import json
import support
import tqdm
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
OPEN_FILE_NAME = 'test'
SAVE_FILE_NAME = 'test'
OPEN_FILE_NAME = r'news/' + OPEN_FILE_NAME + '.txt'
SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.json'

cats = {'media': 1, 'culture': 2, 'sport': 3, 'business': 4, 'science': 5, 'life': 6, 'style': 7, 'economics': 8,
        'forces': 9, 'travel': 10}

def process_text(title, body):
    title, body = support.process_string(title), support.process_string(body)
    stats = []
    #stats.append(<>(title))
    #stats.append( <>(body))
    #stats.append( <>(title, body))
    stats.append(title)
    stats.append(body)
    return stats


def process_file(lines):
    progressbar = tqdm.tqdm(desc='PROCESS', total=len(lines))
    data = []
    tags_names = []
    tags = []
    for line in lines:
        temp = line.split(sep='\t')
        tag, title, body = tuple(temp)
        tags_names.append(tag)
        tags.append(cats[tag])
        data.append(title + ' ' + body)
        progressbar.update(1)
    progressbar.close()
    return data, tags, tags_names


def main():
    with open(OPEN_FILE_NAME, encoding='utf-8') as file:
        news = file.readlines()
    file.close()
    data, tags, tags_names = process_file(news)
    with open(SAVE_FILE_NAME, 'w', encoding='utf-8') as file:
       file.write(json.dumps(data, indent=2, ensure_ascii=False))
    file.close()
    return data, tags

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__n_iter': (10, 50, 80),
}


if __name__ == '__main__':

    data, tags = main()

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data, tags)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))