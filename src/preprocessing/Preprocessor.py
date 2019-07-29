# adapted from https://github.com/whiskeyromeo/CommunityQuestionAnswering

import nltk
import re
from tensorflow.contrib import learn
import numpy as np
from nltk.tokenize import MWETokenizer
import os
import warnings
import collections
import gzip
import pickle


def for_each_question(questions, function):
    for question in questions:
        function(questions[question])
        for relatedQuestion in questions[question]['related']:
            function(questions[question]['related'][relatedQuestion])

class Preprocessor:
    vocab_processor = None

    @staticmethod
    def basic_pipeline(sentences, ngram_replacements, vocab):
        # process text
        print("Preprocessor: replace urls and images")
        sentences = Preprocessor.replaceImagesURLs(sentences, vocab)
        print("Preprocessor: remove punctuation")
        sentences = Preprocessor.removePunctuation(sentences)
        print("Preprocessor: to lower case")
        sentences = Preprocessor.toLowerCase(sentences)
        if not ngram_replacements == {}:
            print('Preprocessor: replace n-grams')
            sentences = Preprocessor.replaceNgrams(sentences, ngram_replacements)
        # todo: include tokenisation in cache to save time when loading
        # print("Preprocessor: split sentence into words")
        # sentences = Preprocessor.tokenize_tweet(sentences)
        return sentences


    @staticmethod
    def create_replacement_dict(vocab):
        '''
        Creates replacement dictionary for ngrams in vocabulary from pretrained word embeddings
        :param vocab: list of vocabulary items from pretrained embeddings
        :return: dictionary with replacements, {} for empty vocabulary
        '''
        replacement_dict = {}
        if not vocab is None:
            for token in vocab:
                match = re.match('[^_\W]+_[^_\W]+([^_\W]+)?',token)
                if match and token.count('_')<=2:
                    to_replace = token.replace('_', ' ')
                    replacement_dict[to_replace] = token
        print('Found {} n-grams in pretrained embeddings'.format(len(replacement_dict.keys())))
        return replacement_dict

    @staticmethod
    def replaceNgrams(sentences, replacement_dict):
        # ToDo: how to speed up?
        def multiple_replace(text, adict):
            # don't match punctuation (look for spaces rather than word boundaries \b)
            rx = re.compile(r'\b%s\b' % r'(?=\s)|(?<=\s)'.join(map(re.escape, adict)))
            # print(rx)
            def one_xlat(match):
                return adict[match.group(0)]
            return rx.sub(one_xlat, text)
        out = []
        for s in sentences:
            replaced = multiple_replace(s, replacement_dict)
            out.append(replaced)
        return out

    @staticmethod
    def replaceImagesURLs(sentences,vocab):
        out = []
        URL_token = None
        IMG_token = None
        URL_tokens = ['<url>','<URL>','URLTOK']  # 'URLTOK' or '<URL>'
        IMG_tokens = ['<pic>','IMG']
        for t in URL_tokens:
            if t in vocab:
                URL_token = t
                break
        if URL_token is None:
            warnings.warn('URL tokens {} not in vocab.'.format(str(URL_tokens)),Warning)
            URL_token = URL_tokens[0]
        for t in IMG_tokens:
            if t in vocab:
                IMG_token = t
                break
        if IMG_token is None:
            warnings.warn('IMG tokens {} not in vocab.'.format(str(IMG_tokens)),Warning)
            IMG_token = IMG_tokens[0]

        for s in sentences:
            s = re.sub('(http://)?www.*?(\s|$)', URL_token+'\\2', s) # URL containing www
            s = re.sub('http://.*?(\s|$)', URL_token+'\\1', s) # URL starting with http
            s = re.sub('\w+?@.+?\\.com.*',URL_token,s) #email
            s = re.sub('\[img.*?\]',IMG_token,s) # image
            out.append(s)
        return out

    @staticmethod
    def removePunctuation(sentences):
        '''
        Remove punctuation from list of strings
        :param sentences: list
        :return: list
        '''
        out = []
        for s in sentences:
            # ToDo: how to omit special tokens?
            # Twitter embeddings retain punctuation and use the following special tokens:
            # <unknown>, <url>, <number>, <allcaps>, <pic>
            # s = re.sub(r'[^\w\s]', ' ', s)
            s = re.sub(r'[^[^a-zA-Z0-9_<>]\s]', ' ', s)
            s = re.sub(r'[\s+]', ' ', s)
            s = re.sub(r' +', ' ', s)  # prevent too much whitespace
            s = s.lstrip().rstrip()
            out.append(s)
        return out

    @staticmethod
    def addBigrams(question):
        question['question_bigram_list'] = list(nltk.bigrams(question['question_words']))
        question['question_bigram_list_nostopwords'] = list(nltk.bigrams(question['question_words_nostopwords']))

    @staticmethod
    def addTrigrams(question):
        question['question_trigram_list'] = list(nltk.trigrams(question['question_words']))
        question['question_trigram_list_nostopwords'] = list(nltk.trigrams(question['question_words_nostopwords']))

    @staticmethod
    def addPartOfSpeech(question):
        question['question_words_pos'] = nltk.pos_tag(question['question_words'])
        question['question_words_pos_nostopwords'] = nltk.pos_tag(question['question_words_nostopwords'])

    @staticmethod
    def stopwordsList():
        stopwords = nltk.corpus.stopwords.words('english')
        return stopwords

    @staticmethod
    def removeStopwords(question):
        stopwords = Preprocessor.stopwordsList()
        return [i for i in question if i not in stopwords]

    @staticmethod
    def removeShortLongWords(sentence):
        return [w for w in sentence if len(w)>1 and len(w)<200]

    @staticmethod
    def tokenize_simple(iterator):
        return [sentence.split(' ') for sentence in iterator]
    @staticmethod
    def tokenize_nltk(iterator):
        return [nltk.word_tokenize(sentence) for sentence in iterator]
    @staticmethod
    def tokenize_tweet(iterator,strip=True):
        # tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        tknzr = nltk.tokenize.TweetTokenizer()
        result = [tknzr.tokenize(sentence) for sentence in iterator]
        if strip:
            result = [[w.replace(" ", "") for w in s] for s in result]
        return result
    @staticmethod
    def tokenize_tf(sentences):
        # from https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/learn/python/learn/preprocessing/text.py
        TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
        return [TOKENIZER_RE.findall(s) for s in sentences]
    # @staticmethod
    # def collapse_multi_word_expressions(sentences):
    #     tokenizer = MWETokenizer()
    #     return [tokenizer.tokenize(s) for s in sentences]
    @staticmethod
    def substitute_stopword(tokenized_sentences,unk_token='UNK'):
        stopwords = Preprocessor.stopwordsList()
        output = []
        for s in tokenized_sentences:
            output.append([unk_token if w in stopwords else w for w in s])
        return output

    @staticmethod
    def removeNonEnglish(question):
        #ToDo
        raise NotImplementedError()

    @staticmethod
    def toLowerCase(sentences):
        out = []
        special_tokens = ['URLTOK', '<PIC>','UNK']
        for s in Preprocessor.tokenize_tweet(sentences):
            sent =[]
            # split sentences in tokens and lowercase except for special tokens
            for w in s:
                if w in special_tokens:
                    sent.append(w)
                else:
                    sent.append(w.lower())
            out.append(' '.join(sent))
        return out

    @staticmethod
    def max_document_length(sentences,tokenizer):
        sentences = tokenizer(sentences)
        return max([len(x) for x in sentences]) # tokenised length of sentence!

    @staticmethod
    def pad_sentences(sentences, max_length,pad_token='<PAD>',tokenized=False):
        '''
        Manually pad sentences with pad_token (to avoid the same representation for <unk> and <pad>)
        :param sentences: 
        :param tokenizer: 
        :param max_length: 
        :param pad_token: 
        :return: 
        '''
        if tokenized:
            tokenized = sentences
            return [(s + [pad_token] * (max_length - len(s))) for s in tokenized]
        else:
            tokenized = Preprocessor.tokenize_tweet(sentences)
            return [' '.join(s + [pad_token] * (max_length - len(s))) for s in tokenized]

    @staticmethod
    def replaceUNK(sentences,old='<UNK_L>', new='<UNK_R>'):
        '''
        Substitute id for old UNK with id for new UNK
        :param sentences: list of sentences with word ids
        :param old: str
        :param new: str
        :return: 
        '''
        # find id of old
        id_old = Preprocessor.word2id[old]
        # find id new
        id_new = Preprocessor.word2id[new]
        # substitute
        sentences[sentences == id_old] = id_new
        return sentences

    @staticmethod
    def reduce_sentence_len(r_tok,max_len):
        '''
        Reduce length of tokenised sentence
        :param r_tok: nested list consisting of tokenised sentences e.g. [['w1','w2'],['w3']]
        :param max_len: maximum length of sentence
        :return: nested list consisting of tokenised sentences, none longer than max_len
        '''
        return [s if len(s) <= max_len else s[:max_len] for s in r_tok]



