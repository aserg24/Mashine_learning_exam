import json
import support
import tqdm
OPEN_FILE_NAME = 'news_test'
OPEN_FILE_NAME = r'news/' + OPEN_FILE_NAME + '.txt'
SAVE_FILE_NAME = 'news_test'
SAVE_FILE_NAME = r'data/' + SAVE_FILE_NAME + '.json'


def process_text(title, body):
    title, body = support.process_string(title), support.process_string(body)
    line = []
    line = '{} {}'.format(' '.join(title), ' '.join(body))
    return line


def process_file(lines):
    progressbar = tqdm.tqdm(desc='PROCESS', total=len(lines))
    data = []
    for line in lines:
        temp = line.split(sep='\t')
        title, body = tuple(temp)
        data.append(process_text(title, body))
        progressbar.update(1)
    progressbar.close()
    return data


def main():
    with open(OPEN_FILE_NAME, encoding='utf-8') as file:
        news = file.readlines()
    file.close()
    data = process_file(news)
    with open(SAVE_FILE_NAME, 'w', encoding='utf-8') as file:
        file.write(json.dumps(data, indent=2, ensure_ascii=False))
    file.close()
    return data


if __name__ == '__main__':
    data = main()