import nltk
from pymorphy2 import MorphAnalyzer
m = MorphAnalyzer()


def process_string(text):
    words = nltk.word_tokenize(text)
    normal_words = []
    for word in words:
        normal_words += process_word(word)
    return normal_words


def process_word(word):
    word_data = m.parse(word)[0]
    if should_be_filtered(word_data):
        return []
    else:
        return [word_data.normal_form]


TagsToBlock = ['PREP', 'PRCL', 'INTJ', 'CONJ', 'NPRO', 'NUMR']
WordsToBlock = ['сказать', 'говорить', 'только', 'другой', 'первый', 'ребята', 'очень', 'большой', 'новый', 'стать',
    'сейчас', 'время', 'человек', 'жизнь', 'каждый', 'самый', 'хотеть', 'здесь', 'теперь', 'пойти', 'город', 'потом',
    'видеть', 'можно', 'много', 'конечно', 'вопрос', 'просто', 'сообщать', 'прийти', 'оказаться', 'который', 'сообщить']


def should_be_filtered(word_data):
    if len(word_data.normal_form) <= 4:
        return True
    elif word_data.normal_form in WordsToBlock:
        return True
    elif word_data.tag.POS in TagsToBlock:
        return True
    else:
        return False
