import pickle
import re
import math
import numpy
import pandas
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import sys
from py4j.java_gateway import JavaGateway, GatewayParameters

enhi_eng_word_dictionary = {}
# dictionary to store all words of hindi(for english-hindi translation) and its corresponding unique code
enhi_hin_word_dictionary = {}
# Translation table of words for english to hindi translation
enhi_translation_table = {}
# dictionary to store all words of english(for hindi-english translation) and its corresponding unique code
hien_eng_word_dictionary = {}
# dictionary to store all words of hindi(for hindi-english translation) and its corresponding unique code
hien_hin_word_dictionary = {}
# Translation table of words for hindi to english translation
hien_translation_table = {}
eng_to_hin_dictionary = {}
word_re = re.compile("\w+")


def gamma(word_lang_1, word_lang_2):
    """
    Returns 1 if given words are in bilingual dictionary
    :rtype: int
    :param word_lang_1: word in language 1
    :param word_lang_2: word in language 2
    :return: 1 or 0
    """
    if len(word_lang_1) == 1:
        return 0
    try:
        possible_translations_of_word_lang_1 = eng_to_hin_dictionary[word_lang_1.lower()] \
                                               + eng_to_hin_dictionary[stemmer.stem(word_lang_1)]
        if word_lang_2 in possible_translations_of_word_lang_1:
            return 1
        else:
            if word_re.match(word_lang_2):
                for root_word in stem_hindi_word(word_lang_2):
                    if root_word in possible_translations_of_word_lang_1:
                        return 1
                return 0
        return 0
    except KeyError:
        return 0


def degree_lang_1(word_lang_1, sentence_lang_2):
    """
    Summation of all gamma of given word of language 1 and all words of sentence of language 2
    :rtype: int
    :param word_lang_1: word of a sentence of language 1
    :param sentence_lang_2: sentence of language 2
    :return: summation of gamma
    """
    temp_sum = 0
    for word_lang_2 in sentence_lang_2.split(" "):
        temp_sum += gamma(word_lang_1, word_lang_2)
    return temp_sum


def degree_lang_2(word_lang_2, sentence_lang_1):
    """
    Summation of all gamma of given word of language 2 and all words of sentence of language 1
    :rtype: int
    :param word_lang_2: word of a sentence of language 2
    :param sentence_lang_1: sentence of language 1
    :return: summation of gamma
    """
    temp_sum = 0
    for word_lang_1 in tokenizer.tokenize(sentence_lang_1):
        temp_sum += gamma(word_lang_1, word_lang_2)
    return temp_sum


def words_having_translation_lang_1(sentence_lang_1, sentence_lang_2):
    """
    Total number of words of sentence belonging to language 1 have translation in sentence belonging to languaage 2
    :rtype: int
    :type sentence_lang_1: str
    :type sentence_lang_2: str
    :param sentence_lang_1: Sentence belonging to language 1
    :param sentence_lang_2: Sentence belonging to language 2
    """
    count = 0
    for word_lang_1 in tokenizer.tokenize(sentence_lang_1):
        if degree_lang_1(word_lang_1, sentence_lang_2) > 0:
            count += 1
    return count


def words_having_translation_lang_2(sentence_lang_2, sentence_lang_1):
    """
    Total number of words of sentence belonging to language 2 have translation in sentence belonging to language 1
    :rtype: int
    :type sentence_lang_2: str
    :type sentence_lang_1: str
    :param sentence_lang_2: Sentence belonging to language 2
    :param sentence_lang_1: Sentence belonging to language 1
    """
    count = 0
    for word_lang_2 in sentence_lang_2.split(" "):
        if degree_lang_2(word_lang_2, sentence_lang_1) > 0:
            count += 1
    return count


def translational_probability(word_lang_1, word_lang_2):
    """
    Fetches translational probability of word of language 1 to word of language 2 from translation table
    :rtype: float
    :param word_lang_1: word belonging to lang_1
    :param word_lang_2: target translated word belonging to language 2
    :return: translational probability
    """
    try:
        unique_code_lang_1 = enhi_eng_word_dictionary[word_lang_1]
        unique_code_lang_2 = enhi_hin_word_dictionary[word_lang_2]
        return enhi_translation_table[unique_code_lang_1][unique_code_lang_2]
    except KeyError:
        return -1


def translational_probability_lang2_to_lang_1(word_lang_2, word_lang_1):
    """
    Fetches translational probability of word of language 2 to word of language 1 from translation table
    :rtype: float
    :param word_lang_2: source word belonging to language 2
    :param word_lang_1: target translated word belonging to lang_1
    :return: translational probability
    """
    try:
        unique_code_lang_1 = hien_eng_word_dictionary[word_lang_1]
        unique_code_lang_2 = hien_hin_word_dictionary[word_lang_2]
        return hien_translation_table[unique_code_lang_2][unique_code_lang_1]
    except KeyError:
        return -1


def ibm_model_1_score(sentence_lang_1, sentence_lang_2):
    # tokenize both sentences to words
    tokenized_sentence_lang_1 = tokenizer.tokenize(sentence_lang_1)
    # inserting NULL token
    tokenized_sentence_lang_1.insert(0, "NULL")
    tokenized_sentence_lang_2 = sentence_lang_2.split(' ')
    # inserting NULL token
    tokenized_sentence_lang_2.insert(0, "NULL")
    best_alignment = []
    temp_translational_score = 1.0
    for word_lang_2 in tokenized_sentence_lang_2[1:]:
        max_translational_prob = 0
        max_i = -1
        # form a pair of word language 1 and word from language 2 and then see its translational probability from giza
        # translational table. Pair having maximum translational probability is selected as translation pair word of
        # lang 1 is added into translated sentence
        for i, word_lang_1 in enumerate(tokenized_sentence_lang_1):
            trans_prob = translational_probability(word_lang_1, word_lang_2)
            if max_translational_prob < trans_prob != -1:
                max_translational_prob = trans_prob
                max_i = i
        if max_translational_prob == 0:
            max_translational_prob = 1
            max_i = 0
        temp_translational_score *= max_translational_prob
        best_alignment.append(max_i)
    # calculating translational probability using ibm model 1
    div = float(pow(len(tokenized_sentence_lang_2), (len(tokenized_sentence_lang_1) - 1)))
    model_1_translational_probability = temp_translational_score / div
    return model_1_translational_probability


def ibm_model_1_score_lang2_to_lang_1(sentence_lang_2, sentence_lang_1):
    # tokenize both sentences to words
    tokenized_sentence_lang_1 = tokenizer.tokenize(sentence_lang_1)
    # inserting NULL token
    tokenized_sentence_lang_1.insert(0, "NULL")
    tokenized_sentence_lang_2 = sentence_lang_2.split(' ')
    # inserting NULL token
    tokenized_sentence_lang_2.insert(0, "NULL")
    # alignments from word of language 2 to words of language 1
    best_alignment = []
    temp_translational_score = 1
    # for each word of language 1
    for word_lang_1 in tokenized_sentence_lang_1[1:]:
        max_translational_prob = 0
        max_i = -1
        # form a pair of word language 1 and word from language 2 and then see its translational probability from giza
        # translational table. Pair having maximum translational probability is selected as translation pair word of
        # lang 2 is added into translated sentence
        for i, word_lang_2 in enumerate(tokenized_sentence_lang_2):
            trans_prob = translational_probability_lang2_to_lang_1(word_lang_2, word_lang_1)
            if max_translational_prob < trans_prob != -1:
                max_translational_prob = trans_prob
                max_i = i
        if max_translational_prob == 0:
            max_translational_prob = 1
            max_i = 0
        temp_translational_score *= max_translational_prob
        best_alignment.append(max_i)
    # calculating translational probability using ibm model 1
    div = pow(len(tokenized_sentence_lang_1), (len(tokenized_sentence_lang_2) - 1))
    model_1_translational_probability = temp_translational_score / div
    return model_1_translational_probability


def stem_hindi_word(hindi_word):
    gateway = JavaGateway(gateway_parameters=GatewayParameters())
    hindi_analyser = gateway.entry_point
    morph_analysis = hindi_analyser.analyse(hindi_word)
    root_words = morph_analysis.getRoots()
    root_words_list = [root_word for root_word in root_words]
    return root_words_list


def word_overlap_score(sentence_lang_1, sentence_lang_2):
    """
    Returns overlap score used to initialize EM algorithm
    :type sentence_lang_2: str
    :type sentence_lang_1: str
    :rtype: float
    :param sentence_lang_1: One Sentence belonging to 1st language 
    :param sentence_lang_2: One Sentence belonging to 2nd language
    """
    return (words_having_translation_lang_1(sentence_lang_1, sentence_lang_2)
            * words_having_translation_lang_2(sentence_lang_2, sentence_lang_1)) / \
           (len(tokenizer.tokenize(sentence_lang_1)) * len(sentence_lang_2.split(" ")))


def strip_newline_char(sentence):
    return sentence.rstrip('\n')


def translation_similarity_score(sentence_lang_1, sentence_lang_2):
    """
    Returns translation similarity score on the basis of IBM model 1 translational probability score from lang 1 to 
    lang 2 and same IBM model 1 score for lang 1 to lang 2
    Formula used is (log(P(s_2|s_1))+log(P(s_1|s_2)))/(l_1+l_2);
    where l_1 is length of sentence belonging to language 1
    and l_2 is length of sentence belonging to language 2

    :rtype: float
    :param sentence_lang_1: sentence belonging to language 1 (source language)
    :param sentence_lang_2: sentence belonging to language 2 (target language)
    :return: translation similarity score given by above formula
    """
    return (math.log(ibm_model_1_score(sentence_lang_1, sentence_lang_2)) +
            math.log(ibm_model_1_score_lang2_to_lang_1(sentence_lang_2, sentence_lang_1))) / \
           (len(sentence_lang_1.split(" ")) + len(sentence_lang_2.split(" ")))


if __name__ == '__main__':
    lang_1_corpus_filename = sys.argv[1]
    lang_2_corpus_filename = sys.argv[2]
    stemmer = PorterStemmer()
    eng_to_hin_dictionary = pickle.load(open('pickle/eng-hin_dictionary.pickle', 'rb'))
    enhi_eng_word_dictionary = pickle.load(open('pickle/eng_word_dictionary.pickle', 'rb'))
    enhi_hin_word_dictionary = pickle.load(open('pickle/hin_word_dictionary.pickle', 'rb'))
    enhi_translation_table = pickle.load(open('pickle/translation_table.pickle', 'rb'))
    hien_eng_word_dictionary = pickle.load(open('pickle/hi-en_eng_word_dictionary.pickle', 'rb'))
    hien_hin_word_dictionary = pickle.load(open('pickle/hi-en_hin_word_dictionary.pickle', 'rb'))
    hien_translation_table = pickle.load(open('pickle/hi-en_translation_table.pickle', 'rb'))
    tokenizer = RegexpTokenizer(r'\w+')
    best_aligned = []
    alignment_score = {}
    count_conditional = {}
    with open(lang_1_corpus_filename, 'r') as eng_sentences:
        with open(lang_2_corpus_filename, 'r') as hin_sentences:
            hin_sentences = hin_sentences.readlines()
            eng_sentences = eng_sentences.readlines()
            hin_sentences = list(map(strip_newline_char, hin_sentences))
            eng_sentences = list(map(strip_newline_char, eng_sentences))
            for en_index, eng_sentence in enumerate(eng_sentences):
                alignment_score[en_index] = {}
                count_conditional[en_index] = {}
                for hi_index, hin_sentence in enumerate(hin_sentences):
                    alignment_score[en_index][hi_index] = word_overlap_score(eng_sentence, hin_sentence)
                    count_conditional[en_index][hi_index] = 0.0
            alignment_score = pandas.DataFrame.from_dict(alignment_score)
            count_conditional = pandas.DataFrame.from_dict(count_conditional)
            total = pandas.Series(data=numpy.zeros(shape=len(eng_sentences)))
            s_total = pandas.Series(data=numpy.zeros(shape=len(hin_sentences)))
            count = 0
            while count <= 1:
                for hi_index, hin_sentence in enumerate(hin_sentences):
                    s_total[hi_index] = 0.0
                    for en_index, eng_sentence in enumerate(eng_sentences):
                        s_total[hi_index] += alignment_score[en_index][hi_index]
                for hi_index, hin_sentence in enumerate(hin_sentences):
                    for en_index, eng_sentence in enumerate(eng_sentences):
                        count_conditional.loc[hi_index, en_index] += \
                            alignment_score[en_index][hi_index] / s_total[en_index]
                        total[en_index] += alignment_score[en_index][hi_index] / s_total[en_index]
                print("count_conditional")
                print(count_conditional)
                print("total")
                print(total)
                for en_index, eng_sentence in enumerate(eng_sentences):
                    for hi_index, hin_sentence in enumerate(hin_sentences):
                        alignment_score.loc[hi_index, en_index] = count_conditional[en_index][hi_index] \
                                                                  / total[en_index]
                print(alignment_score)
                count += 1
            threshold = -5
            maxi = alignment_score.idxmax(axis=0)
            with open("output_1/final_corpus" + str(threshold) + ".hi", 'w') as parallel_hindi_file:
                with open('output_1/final_corpus' + str(threshold) + '.en', 'w') as parallel_english_file:
                    for hi_index, en_index in enumerate(maxi):
                        if translation_similarity_score(hin_sentences[hi_index], eng_sentences[en_index]) > threshold:
                            parallel_hindi_file.write(hin_sentences[hi_index] + '\n')
                            parallel_english_file.write(eng_sentences[en_index] + '\n')
