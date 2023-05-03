import os
import pickle
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
import collections
import math
from uroman import Uroman
import pandas as pd

pbc_info = pd.read_csv('./pbc_table.csv', converters={"language_code": str})

# see if the verses need to be romanized
def requires_romanize(file_name):
    results = pbc_info.loc[pbc_info['file_name']==file_name]['script_code'].values
    if len(results) == 0:
        print('The file does not exits!')
        return True
    else:
        script_code = results[0]
    # print(script_code)
    if script_code == 'Latn' or script_code == 'Cyrl':
        return False
    else:
        return True

def get_full_language_name(code):
    results = pbc_info.loc[pbc_info['language_code']==code]['language_name'].values
    if len(results) == 0:
        print('The language code is not included!')
        return code
    else:
        return results[0]

# romanize the verses 
def romanize(verses, lang=None):
    uroman_ = Uroman()
    romanized_sents = uroman_.romanize(verses, lang=lang)
    return romanized_sents


# load from the stored path
def load_statistics(temp_dir, store_name):
    print(temp_dir + '/' + store_name + '.pickle')
    if os.path.exists(temp_dir + '/' + store_name + '.pickle'):
        with open(temp_dir + '/' + store_name + '.pickle', 'rb') as handle:
            result = pickle.load(handle)
        return result
    else:
        print("File does not exist.")
    # else:
    #     raise FileNotFoundError


# load the statistics given the name
def load_result(concept_string, temp_dir=None):
    if temp_dir is None:
        temp_dir = './results'
    store_name = concept_string + '/concept.results'
    return load_statistics(temp_dir, store_name)


# remove the $ in the n-gram
def simplify_string(string):
    string = string[1:] if string[0] == '$' else string
    string = string[:-1] if string[-1] == '$' else string
    return string.replace('$', ' ')


# check undesired n-grams
def check_ngram_validity(ngram):
    key = ngram
    if '!' in key:
        return False
    elif '?' in key:
        return False
    elif '.' in key:
        return False
    elif ',' in key:
        return False
    elif '[' in key:
        return False
    elif ']' in key:
        return False
    elif ';' in key:
        return False
    elif ':' in key:
        return False
    elif '“' in key:
        return False
    elif '′' in key:
        return False
    elif '·' in key:
        return False
    elif '$$' in key:
        return False
    return True


# remove undesired n-grams
def filter_translations(translation_list):
    filtered_translation_list = []
    for translation in translation_list:
        if check_ngram_validity(translation):
            filtered_translation_list.append(translation)
    return filtered_translation_list


# create the list of the associated concepts in order of how many languages are involved
def create_concept_map(result, search_strings):

    # the first position stores the number of language which contains the n-gram in reverse search
    # the second position stores the total number of occurrence of this string in all languages
    # the last position stores the language ISO 639-3 codes of the first position
    counter = defaultdict(lambda: [0, 0, []])

    # the first is the number of languages, the second is the total occurrences, the list to store languages
    for key, value in result.items():
        lang = key.split('-')[-1]
        count_flag = True  # for each language only count once the prototype (which is the center)

        # if no FP
        if len(value['tgt_string_translations']) == 0:
            counter['prototype'][0] += 1  # number of language
            counter['prototype'][1] += value['statistics1'][1][2][0][0]
            if lang not in counter['prototype']:
                counter['prototype'][2].append(lang)
                counter['prototype'][2] = sorted(counter['prototype'][2])
            continue

        # if there are FP
        temp_dict = dict(value['statistics2'])
        for tr in filter_translations(value['tgt_string_translations']):
            if simplify_string(tr) in search_strings:
                counter['prototype'][1] += temp_dict[tr][2][0][0]
                if lang not in counter['prototype'][2]:
                    counter['prototype'][2].append(lang)
                    counter['prototype'][2] = sorted(counter['prototype'][2])
                if count_flag:
                    counter['prototype'][0] += 1
                    count_flag = False
            else:
                counter[tr][0] += 1
                counter[tr][1] += temp_dict[tr][2][0][0]
                if lang not in counter[tr]:
                    counter[tr][2].append(lang)
                    counter[tr][2] = sorted(counter[tr][2])

    counter = Counter(counter)
    return counter.most_common()


# return a dictionary of the {ngram1: {expanded_str1, expanded_str2}, ngram2: {}, ...}
def create_concept_expanded_dict(result):
    expanded_dict = defaultdict(lambda: set())
    for key, value in result.items():
        temp_dict = dict(value['statistics2'])
        for tr in value['tgt_string_translations']:
            temp_set = set([key for key, value in dict(temp_dict[tr][4]).items()])
            # print(temp_set)
            expanded_dict[tr].update(temp_set)
    return dict(expanded_dict)


# lemmatize the concept and merge them if the lemmas are the same
def lemmatize_concepts(concept_list, expanded_dict):
    igore_indeces = []
    new_concept_list = []
    concept_forms = {}
    new_concept_list.append(concept_list[0])
    for i in range(1, len(concept_list)):

        if i in igore_indeces:
            continue

        # use the longest string
        expanded_strings = list(expanded_dict[concept_list[i][0]])
        if len(expanded_strings) == 1:
            concept_list[i] = (expanded_strings[0], concept_list[i][1])
        else:
            # if the expanded strings all have the same lemma, we use the lemma
            flag = True
            lemma = lemmatize(simplify_string(expanded_strings[0]))
            for string in expanded_strings[1:]:
                if lemma != lemmatize(simplify_string(string)):
                    flag = False
                    break
            if flag:
                # print(lemma)
                concept_list[i] = (lemma, concept_list[i][1])

        concept1 = simplify_string(concept_list[i][0])
        lemma1 = lemmatize(concept1)
        if lemma1 in concept_forms:
            continue
        flag = True  # indicator of whether the first time to find a same lemma
        flag_others = False  # indicator of whether there is other words with the same lemma
        for j in range(i+1, len(concept_list)):
            concept2 = simplify_string(concept_list[j][0])
            lemma2 = lemmatize(concept2)
            if lemma1 == lemma2:
                # print(concept1, concept2)
                igore_indeces.append(j)  # mark j as it already being integrated
                if flag:
                    common_langs = sorted(list(set(concept_list[i][1][2]).union(set(concept_list[j][1][2]))))
                    new_concept_list.append((lemma1,
                                                  [len(common_langs),
                                                   concept_list[i][1][1] + concept_list[j][1][1],
                                                   common_langs]))
                    concept_forms[lemma1] = len(new_concept_list) - 1  # the index in that list
                    flag = False
                else:
                    # not the first time
                    common_langs = sorted(list(set(new_concept_list[concept_forms[lemma1]][1][2]).union(
                                        set(concept_list[j][1][2]))))
                    new_concept_list[concept_forms[lemma1]] = (lemma1,
                                                          [len(common_langs),
                                                           new_concept_list[concept_forms[lemma1]][1][1] +
                                                           concept_list[j][1][1],
                                                           common_langs])
                flag_others = True
        if flag_others:
            pass
        else:
            new_concept_list.append(concept_list[i])
    new_concept_list = sorted(new_concept_list, key=lambda x: x[1][0], reverse=True)
    return new_concept_list


# filter those associated concepts of which less than two languages have the associations
# and we set a maximum number of associations
def filter_outlier_concepts(concept_list, max_num=100):
    filtered_concept_list = []
    for concept in concept_list:
        assert concept[1][0] == len(concept[1][2])
        if concept[1][0] < 2:
            pass
        else:
            filtered_concept_list.append(concept)
    return filtered_concept_list[:max_num]


def lemmatize(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


# define stability of the concept
def stability(concept_list):
    """
    the stability is the ratio of:
    (1) the number of reverse matches back to the search strings (source concepts)
    and
    (2) the total number of all reverse matches which occur at least in two languages
    """
    return round(concept_list[0][1][1] /
                 sum([concept[1][1] for concept in concept_list
                      if concept[1][0] >= 2]), 2)


# for creating representations, we need to work on the server
# create a normalized one-hot vector given a result of one concept
def create_representation(lang, value, filtered_concept_list, search_strings):
    concept_dict_con_to_int = dict(zip([concept[0] for concept in filtered_concept_list],
                                       [i for i in range(len(filtered_concept_list))]))
    one_hot_vector = np.zeros((len(filtered_concept_list), 1))
    selected_concepts = [con[0] for con in filtered_concept_list]
    # print(selected_concepts)
    if len(value['tgt_string_translations']) == 0:
        one_hot_vector[0] = 1  # number of language
    else:
        total_num = 0
        counter_dict = defaultdict(int)
        temp_dict = dict(value['statistics2'])
        for tr in value['tgt_string_translations']:
            # print(tr)
            total_num += temp_dict[tr][2][0][0]
            if simplify_string(tr) in search_strings:
                counter_dict['prototype'] += temp_dict[tr][2][0][0]
            else:
                lemma = lemmatize(simplify_string(tr))
                if tr in selected_concepts:
                    counter_dict[tr] += temp_dict[tr][2][0][0]
                elif lemma in selected_concepts:
                    counter_dict[lemma] += temp_dict[tr][2][0][0]
                else:
                    expanded_strings = list(temp_dict[tr][4].keys())
                    # print(expanded_strings)
                    if len(expanded_strings) == 1:
                        # if the length of the expanded string is just 1, we use the lemma if the lemma is in the list
                        # or the word itself
                        if lemmatize(simplify_string(expanded_strings[0])) in selected_concepts:
                            counter_dict[lemmatize(simplify_string(expanded_strings[0]))] += temp_dict[tr][2][0][0]
                        else:
                            counter_dict[expanded_strings[0]] += temp_dict[tr][2][0][0]
                    else:
                        # if the expanded strings all have the same lemma, we use the lemma
                        flag = True
                        lemma = lemmatize(simplify_string(expanded_strings[0]))
                        for string in expanded_strings[1:]:
                            if lemma != lemmatize(simplify_string(string)):
                                flag = False
                                break
                        if flag:
                            counter_dict[lemma] += temp_dict[tr][2][0][0]
                        else:
                            counter_dict[tr] += temp_dict[tr][2][0][0]
        # print(lang, counter_dict, value['tgt_string_translations'])
        for key, value in counter_dict.items():
            if key in concept_dict_con_to_int:
                one_hot_vector[concept_dict_con_to_int[key]] = value / total_num
    return one_hot_vector


# concatenate multiple one-hot vectors of different concepts
def create_representations_for_group_of_concepts(concept_strings, maximum_ngram=100, result_dict=None, temp_dir=None):
    representations = defaultdict(lambda: [])
    filtered_representations = {}
    for concept_string in concept_strings:
        search_strings = [simplify_string(w) for w in concept_string.split('-')]
        result = load_result(concept_string, temp_dir) if temp_dir is not None else load_result(concept_string)
        if result_dict is None:
            concept_list = create_concept_map(result, search_strings)
            expanded_dict = create_concept_expanded_dict(result)
        else:
            concept_list, expanded_dict = result_dict[concept_string]
        concept_list = lemmatize_concepts(concept_list, expanded_dict)
        filtered_concept_list = filter_outlier_concepts(concept_list, max_num=maximum_ngram)
        selected_concepts = [con[0] for con in filtered_concept_list]
        # print(selected_concepts)
        for lang, value in result.items():
            lang = lang.split('-')[-1]
            representations[lang].append(create_representation(lang, value, filtered_concept_list, search_strings))
    for lang, value in representations.items():
        if len(value) != len(concept_strings):
            continue
        else:
            filtered_representations[lang] = np.concatenate([vec for vec in value], axis=0)

    return filtered_representations


# compute the cosine similarity between any two representations
def get_similarity(lang1, lang2, representations):
    cos_sim = cosine_similarity(representations[lang1].reshape(1, -1), representations[lang2].reshape(1, -1))
    return cos_sim[0][0]
