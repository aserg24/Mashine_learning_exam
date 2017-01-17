import json
OPEN_FILE_NAME = 'news_train'
SAVE_FILE_NAME = 'news_train_1'
OPEN_FILE_NAME = r'data/' + OPEN_FILE_NAME + '.json'
SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.json'

if __name__ == '__main__':
    with open(OPEN_FILE_NAME, encoding='utf-8') as file:
        news = json.loads(file.read())
    file.close()
    data = []
    for i in news:
        data.append('{} {}'.format(' '.join(i[0]), ' '.join(i[1])))
    with open(SAVE_FILE_NAME, 'w', encoding='utf-8') as file:
       file.write(json.dumps(data, indent=2, ensure_ascii=False))
    file.close()

cats = {'media': 1, 'culture': 2, 'sport': 3, 'business': 4, 'science': 5, 'life': 6, 'style': 7, 'economics': 8,
            'forces': 9, 'travel': 10}


OPEN_FILE_NAME = 'news_train'
SAVE_FILE_NAME = 'tags'
OPEN_FILE_NAME = r'news/' + OPEN_FILE_NAME + '.txt'
SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.json'
if __name__ == '__main__':
    with open(OPEN_FILE_NAME, encoding='utf-8') as file:
        news = file.readlines()
    file.close()
    tags = []
    for line in news:
        temp = line.split(sep='\t')
        tag, title, body = temp
        tags.append(cats[tag])
    with open(SAVE_FILE_NAME, 'w', encoding='utf-8') as file:
       file.write(json.dumps(tags, indent=2, ensure_ascii=False))
    file.close()