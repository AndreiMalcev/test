from nltk.tokenize import wordpunct_tokenize
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker


nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stop_words.update(['?', ','])
spell = SpellChecker()
spell.word_frequency.load_text_file('russian.txt', encoding='windows-1251')


def remove_stop_words(sentence):
    new_sentence = ''
    for word in wordpunct_tokenize(sentence):
        if word not in stop_words and not word.isdigit():
            new_sentence += spell.correction(word) + ' '
    return new_sentence[:len(new_sentence) - 1].lower()


def delete_words_x(x):
    for i, sentence in enumerate(x):
        x[i] = remove_stop_words(sentence)
