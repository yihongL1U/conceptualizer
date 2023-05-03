import numpy as np
from parse_pbc import PBC_filenames
import os
import argparse
from scipy.stats import chisquare, chi2_contingency
import itertools
import multiprocessing
import time
import re
import editdistance
import pickle
from utils import *
import markdown
from Levenshtein import ratio
import collections


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
	"""
	Parse boolean arguments from the command line.
	"""
	if s.lower() in FALSY_STRINGS:
		return False
	elif s.lower() in TRUTHY_STRINGS:
		return True
	else:
		raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def get_parser():
	"""
	Generate a parameters parser.
	"""
	parser = argparse.ArgumentParser(description="concept alignment")

	# main parameters
	parser.add_argument("--src_lang", type=str, help="the concept in source languages", default='eng')
	parser.add_argument("--src_string", type=str, help="the source languages considered")
	parser.add_argument("--tgt_langs", type=str, help="the languguages considered", default='fra-deu-zho')
	parser.add_argument("--target_num", type=int, help="how many (sub)words to store in decreasing order ", default=5)
	parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings", default=True)
	parser.add_argument("--minimum_ngram_len", type=int, help="Whether ignore the uppercase when comparing the strings", default=1)
	parser.add_argument("--use_edition_most", type=bool_flag, help="whether use the edition which has the most verses", default=False)
	parser.add_argument("--multiprocessing", type=bool_flag, help="Whether multiprocessing is used", default=True)
	parser.add_argument("--store_dir", type=str, help="the drectory where the results will be stored", 
						default='/mounts/Users/student/yihong/Documents/concept_align/results')
	parser.add_argument("--print_dir", type=str, help="the drectory where the print files will be stored", 
						default='/mounts/Users/student/yihong/Documents/concept_align/printpdf')
	parser.add_argument("--threshold", type=float, help='the threshold for stoping the recursive and reverse search', default=0.85)
	parser.add_argument("--window_size", type=int, help='Slicing Window size for creating combinations of words', default=1)
	parser.add_argument("--ngram_min_len", type=int, help='minimum length of the ngram', default=1)
	parser.add_argument("--ngram_max_len", type=int, help='minimum length of the ngram', default=8)
	parser.add_argument("--make_annotation", type=bool_flag, help='whether print the pdfs to make annotations', default=False)
	return parser


# only for english
def sub_string(s: str, ngrams2count ,minimum_ngram_len=3):
	# mark the begining and the ending of the original token 
	results = []
	for x in range(len(s)):
		for i in range(len(s) - x):
				pre = ''
				suf = ''
				if i == 0:
					 pre = '$'
				if i + x == len(s) - 1:
					 suf = '$'
				if len(s[i:i + x + 1]) < minimum_ngram_len:
					continue
				ngrams2count[pre + s[i:i + x + 1] + suf] += 1
	ngrams2count[s] += 1
	ngrams2count['$' + s] += 1
	ngrams2count[s + '$'] += 1
	# if combination of multiple words are used, we also need to replace ' ' with '$'
	return ngrams2count

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


def obtain_ngram(sent, ngrams2count, ngram_min_len, ngram_max_len):
	sent = ('$' + sent + '$').replace(' ', '$').lower()
	for i in range(len(sent)):
		for j in range(i + ngram_min_len, i + ngram_max_len + 1):
			if j > len(sent) - 1:
				break
			if check_ngram_validity(sent[i:j]):
				ngrams2count[sent[i:j]] += 1
	return ngrams2count

# for each sentence, we create a slicing window to generate combination of words
# window_size = 1: whitespace-split token
# window_size = len(sent), the whole sentence (in this case, we'll generate all n-grams)
# only for english
def obtain_word_ngram(sent, ngrams2count, window_size=1, minimum_ngram_len=3):
	word_sequence = sent.split(' ')
	for i in range(len(word_sequence)):
		temp = ''
		if len(word_sequence) - i < window_size:
			for k in range(len(word_sequence) - i):
				if k == len(word_sequence) - i - 1:
					temp += word_sequence[i+k]
				else:
					temp += word_sequence[i+k] + ' '
		else:
			for k in range(window_size):
				if k == window_size - 1:
					temp += word_sequence[i+k]
				else:
					temp += word_sequence[i+k] + ' '
		ngrams2count = sub_string(temp, ngrams2count, minimum_ngram_len)
	# # print(ngrams2count)
	return ngrams2count


def read_verses(version):
	f = open(version,'r', encoding="utf-8")
	contents = []
	verseIDs = []
	last_verse = ''
	empty_flag = [False]
	for line in f.readlines():
		if line[0] == "#":
			continue
		parts = line.strip().split('\t')
		if len(parts) == 2:
			if empty_flag[-1] == True:
				for i in range(len(empty_flag)):
					contents[-(1+i)] = contents[-(1+i)] + ' ||| ' + parts[1]
				empty_flag = [False]
			verseIDs.append(parts[0])
			contents.append(parts[1])
			last_verse = parts[1]
		elif len(parts) == 1:
			# if it is an empty line, we merge the last and the next verse
			verseIDs.append(parts[0])
			contents.append(last_verse)
			if len(empty_flag) == 1:
				empty_flag = [True]
			else:
				empty_flag.append(True)
	f.close()
	return verseIDs, contents

# for finding the verse that has the most verses
# find the bible version for a given language
def find_version_most(language_code):
	if language_code == 'eng':
		return '/nfs/datc/pbc/eng-x-bible-newworld1984.txt', 'eng-x-bible-newworld1984.txt'
	versions = []
	for filename in os.listdir('/nfs/datc/pbc/'):
		fp = os.path.join('/nfs/datc/pbc/', filename)
		if os.path.isfile(fp) and filename[:3] == language_code:
			versions.append(fp)
	if len(versions) == 0:
		raise ValueError
	elif len(versions) == 1:
		parts = versions[0].split('/')
		return versions[0], parts[-1]
	else:
		# check if there is a newworld version for this language:
		newworld = []
		candidate = ('', 0)
		for v in versions:
			if 'newworld' in v:
				newworld.append(v)
			f = open(v, 'r', encoding="utf-8")
			length_f = len(f.readlines())
			f.close()
			if length_f > candidate[1]:
				candidate = (v, length_f)
		if len(newworld) == 0:
			parts = candidate[0].split('/')
			return candidate[0], parts[-1]
		elif len(newworld) == 1:
			return newworld[0], newworld[0].split('/')[-1]
		else:
			# multiple new world is available
			for v in newworld:
				f = open(v, 'r', encoding="utf-8")
				length_f = len(f.readlines())
				f.close()
				if length_f > candidate[1]:
					candidate = (v, length_f)
			return candidate[0], candidate[0].split('/')[-1]

# for finding the new testaments
# find the bible version for a given language
def find_version_new(language_code):
	if language_code == 'eng':
		return '/nfs/datc/pbc/eng-x-bible-world.txt', 'eng-x-bible-world.txt'
	else:
		versions = []
		for filename in os.listdir('/nfs/datc/pbc/'):
			fp = os.path.join('/nfs/datc/pbc/', filename)
			if os.path.isfile(fp) and filename[:3] == language_code:
				versions.append(fp)
		versions = sorted(versions)
		if len(versions) == 0:
			raise ValueError
		elif len(versions) == 1:
			parts = versions[0].split('/')
			return versions[0], parts[-1]
		else:
			for v in versions:
				f = open(v, 'r', encoding="utf-8")
				length_f = len(f.readlines())
				f.close()
				if length_f < 10000:
					parts = v.split('/')
					return v, parts[-1]
			parts = versions[0].split('/')
			return versions[0], parts[-1]


# find the parallel contents that english and tgt language
def obtain_parallel_contents(src_version, tgt_version):
	src_verseIDs, src_contents = read_verses(src_version)
	tgt_verseIDs, tgt_contents = read_verses(tgt_version)
	common_verseIDs = list(set(src_verseIDs).intersection(set(tgt_verseIDs)))
	common_verseIDs = sorted(common_verseIDs) 
	src_index = [src_verseIDs.index(ID) for ID in common_verseIDs]
	tgt_index = [tgt_verseIDs.index(ID) for ID in common_verseIDs]
	src_parallel_contents = [src_contents[i] for i in src_index]
	tgt_parallel_contents = [tgt_contents[i] for i in tgt_index]
	return src_parallel_contents, tgt_parallel_contents, common_verseIDs


def find_candidates_and_verses_indeces(src_string, src_parallel_contents, tgt_parallel_contents, minimum_ngram_len, 
									  ignore_case=True, ngram_min_len=1, ngram_max_len=8, find_word_ngram=False, window_size=1):
	verse_indeces = [] # verses indeces that have s_e

	ngrams2count = collections.defaultdict(int)

	if isinstance(src_string, str):
		src_string = src_string.lower() if ignore_case else src_string
	else:
		src_string = [string.lower() for string in src_string]

	for i in range(len(src_parallel_contents)):
		src_sent = ('$' + src_parallel_contents[i] + '$').replace(' ', '$').lower() if ignore_case else ('$' + src_parallel_contents[i] + '$').replace(' ', '$')

		if find_word_ngram:
			if isinstance(src_string, str):
				if src_string in src_sent:
					verse_indeces.append(i)
					ngrams2count = obtain_word_ngram(tgt_parallel_contents[i], ngrams2count, window_size=window_size, minimum_ngram_len=minimum_ngram_len)
			else:
				for string in src_string:
					if string in src_sent:
						verse_indeces.append(i)
						ngrams2count = obtain_word_ngram(tgt_parallel_contents[i], ngrams2count, window_size=window_size, minimum_ngram_len=minimum_ngram_len)
						break
		else:
			if isinstance(src_string, str):
				if src_string in src_sent:
					verse_indeces.append(i)
					ngrams2count = obtain_ngram(tgt_parallel_contents[i], ngrams2count, ngram_min_len, ngram_max_len)
			else:
				for string in src_string:
					if string in src_sent:
						verse_indeces.append(i)
						ngrams2count = obtain_ngram(tgt_parallel_contents[i], ngrams2count, ngram_min_len, ngram_max_len)
						break

	if len(verse_indeces) == 0:
		print('No string has been found in the src language!')
	# TO DO
	# adapting the minimum value to be 
	minimum_value = min(max(1, len(verse_indeces) // 10), 3) # the value will be between [1, 5]
	if find_word_ngram:
		minimum_value = min(max(1, len(verse_indeces) // 10), 3) # the value will be between [1, 5]
		# minimum_value = 2

	maximum_value = min(len(src_parallel_contents) // 10, 10000)
	# remove too frequent and too not frequent ngrams
	candidate_translations = [key for key, value in ngrams2count.items() if value >= minimum_value and value <= maximum_value]
	# candidate_translations = [key for key, value in ngrams2count.items() if value >=2]
	return candidate_translations, verse_indeces


def compute_statistics(candidate_translations, tgt_parallel_contents, verse_indeces, common_verseIDs, ignore_indeces, is_reverse=False, is_first_search=False):

	if is_reverse or is_first_search: # (when the source search string is there, we do not want to remove it)
		minimum_value = 1
	else:
		minimum_value = min(max(1, len(verse_indeces) // 10), 3)
	statistics = {}
	for i in range(len(tgt_parallel_contents)):
		if i in ignore_indeces:
			continue
		sent = ('$' + tgt_parallel_contents[i] + '$').replace(' ', '$').lower()
		for test_candidate in candidate_translations:
			if test_candidate not in statistics:
				stat = 0
				p = None
				temp = np.zeros((2, 2), dtype=int)
				verseIDs = [[], [], []]
				tp_ngram2words = collections.defaultdict(int)
				fp_ngram2words = collections.defaultdict(int)
				statistics[test_candidate] = [stat, p, temp, verseIDs, tp_ngram2words, fp_ngram2words]
			if i in verse_indeces: # FREQ set
				begin_indx = sent.find(test_candidate)
				end_indx = begin_indx + len(test_candidate) - 1
				if begin_indx != -1: # found the candidate
					statistics[test_candidate][2][0][0] += 1
					statistics[test_candidate][3][0].append(common_verseIDs[i]) # TP (s_string occur and t_string occur)

					# find the full words
					while sent[begin_indx] != '$':
						begin_indx = begin_indx - 1
					while sent[end_indx] != '$':
						end_indx += 1
					statistics[test_candidate][4][sent[begin_indx:end_indx+1]] += 1
				else:
				 	statistics[test_candidate][2][1][0] += 1
				 	statistics[test_candidate][3][2].append(common_verseIDs[i]) # FN (s_string occur but t_string does not occur)
			else: # ZERO set
				begin_indx = sent.find(test_candidate)
				end_indx = begin_indx + len(test_candidate) - 1
				if begin_indx != -1: # found the candidate
					statistics[test_candidate][2][0][1] += 1
					statistics[test_candidate][3][1].append(common_verseIDs[i]) # FP (t_string occur but s_string does not occur)

					# find the full words
					while sent[begin_indx] != '$':
						begin_indx = begin_indx - 1
					while sent[end_indx] != '$':
						end_indx += 1
					statistics[test_candidate][5][sent[begin_indx:end_indx+1]] += 1
				else:
					statistics[test_candidate][2][1][1] += 1

	for test_candidate in candidate_translations:
		# there is a possibility that some very frequent n-gram feature, e.g., 'a' and 'e' can occurr in every verse
		# so we need to filter them in case we got an error when computing the chisquare scores
		if statistics[test_candidate][2][0][0] < minimum_value:
			# if the string does not occur in the rest verses in recursive search
			statistics.pop(test_candidate)
			continue
		if is_reverse == False or is_first_search == False:
			# remove the n-gram that has so many false positives in recursive search
			if statistics[test_candidate][2][0][1] // 20 > statistics[test_candidate][2][0][0]:
				statistics.pop(test_candidate)
				continue
		if statistics[test_candidate][2][1][0] == 0 and statistics[test_candidate][2][1][1] == 0: # filter very common n-grams
			statistics.pop(test_candidate)
			continue
		elif statistics[test_candidate][2][0][1] == 0 and statistics[test_candidate][2][1][1] == 0: # prevent possible errors when the search string is bad
			statistics.pop(test_candidate)
			continue
		elif statistics[test_candidate][2][0][0] == 0 and statistics[test_candidate][2][0][1] == 0: # filter out possible errors for 1-to-n problem
			statistics.pop(test_candidate)
			continue
		elif statistics[test_candidate][2][0][1] > (np.sum(statistics[test_candidate][2]) // 10): # filter out strings that occur to many times(like 1-gram char, 'e', ...)
			statistics.pop(test_candidate)
			continue
		stat, p, dof, expected = chi2_contingency(statistics[test_candidate][2], correction=True)
		statistics[test_candidate] = (stat, p, statistics[test_candidate][2], statistics[test_candidate][3], 
									  statistics[test_candidate][4], statistics[test_candidate][5])
	return statistics

# filtering the overlapped strings with same statistics
# maybe just when proposing the most significant strings
def filter_string(dict_for_lang, minimum_len=None):
	results = {}
	for key1, value1 in dict_for_lang.items():
		flag = True
		# this is used to filter out some too short strings (but keep in mind the spacial case of chinese and japanese)
		if minimum_len is not None:
			if len(key1) < minimum_len:
				continue
		for key2, value2 in dict_for_lang.items():
				if key1 == key2:
					continue
				elif key1 in key2:
					if dict_for_lang[key1][0] == dict_for_lang[key2][0]:
						flag = False
						break
				else:
					pass
		if flag:
			results[key1] = dict_for_lang[key1]
	return results


def clean_results(results, num):
	# results is a dictionary {'a': chisquare score, 'b': ...}
	# first using chisquare as metirc, then using n00 as metric to sort.
	items = sorted(results.items(), key=lambda item: (item[1][0], item[1][2][0][0], -item[1][2][0][1]), reverse=True)
	return items[:num]


def find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, id_list):
	results = []
	for current_id in id_list:
		index = parallel_verse_IDs.index(current_id)
		results.append((src_sentences[index], tgt_sentences[index]))
	return results

def highlight_string(verse_pair, src_string, tgt_string):
	src_reg = []
	tgt_reg = []

	is_list = False
	if isinstance(src_string, list):
		is_list = True
		for string in src_string:
			src_reg.append(re.compile(re.escape(string), re.IGNORECASE))
	else:
		src_reg = re.compile(re.escape(src_string), re.IGNORECASE)

	tgt_is_list = False
	if isinstance(tgt_string, list):
		tgt_is_list = True
		for string in tgt_string:
			tgt_reg.append(re.compile(re.escape(string), re.IGNORECASE))
	else:
		tgt_reg = re.compile(re.escape(tgt_string), re.IGNORECASE)

	verse_src = ('$' + verse_pair[0] + '$').replace(' ', '$')
	verse_tgt = ('$' + verse_pair[1] + '$').replace(' ', '$')	

	src_pres = ['**' for i in range(len(src_string))] if is_list else '**'
	src_subs = ['**' for i in range(len(src_string))] if is_list else '**'
	if is_list:
		for i in range(len(src_string)):
			if src_string[i][0] == '$':
				src_pres[i] = ' **'
			if src_string[i][-1] == '$':
				src_subs[i] = '** '
	else:
		if src_string[0] == '$':
			src_pres = ' **'
		if src_string[-1] == '$':
			src_subs = '** '

	tgt_pres = ['**' for i in range(len(tgt_string))] if tgt_is_list else '**'
	tgt_subs = ['**' for i in range(len(tgt_string))] if tgt_is_list else '**'
	if tgt_is_list:
		for i in range(len(tgt_string)):
			if tgt_string[i][0] == '$':
				tgt_pres[i] = ' **'
			if tgt_string[i][-1] == '$':
				tgt_subs[i] = '** '
	else:
		if tgt_string[0] == '$':
			tgt_pres = ' **'
		if tgt_string[-1] == '$':
			tgt_subs = '** '


	if is_list:
		for i in range(len(src_string)):
			temp = src_string[i].replace('$', ' ')
			temp = temp[1:] if temp[0] == ' ' else temp
			temp = temp[:-1] if temp[-1] == ' ' else temp
			verse_src = src_reg[i].sub(src_pres[i] + temp.replace('$', '') + src_subs[i], verse_src)
	else:
		temp = src_string.replace('$', ' ')
		temp = temp[1:] if temp[0] == ' ' else temp
		temp = temp[:-1] if temp[-1] == ' ' else temp
		verse_src = src_reg.sub(src_pres + temp.replace('$', '') + src_subs, verse_src)

	if tgt_is_list:
		for i in range(len(tgt_string)):
			temp = tgt_string[i].replace('$', ' ')
			temp = temp[1:] if temp[0] == ' ' else temp
			temp = temp[:-1] if temp[-1] == ' ' else temp
			verse_tgt = tgt_reg[i].sub(tgt_pres[i] + temp.replace('$', '') + tgt_subs[i], verse_tgt)
	else:
		# we need to change "$" to " " if "$" is between two ngrams
		temp = tgt_string.replace('$', ' ')
		temp = temp[1:] if temp[0] == ' ' else temp
		temp = temp[:-1] if temp[-1] == ' ' else temp
		verse_tgt = tgt_reg.sub(tgt_pres + temp.replace('$', '') + tgt_subs, verse_tgt)

	verse_pair = (verse_src[1:-1].replace('$', ' '),  verse_tgt[1:-1].replace('$', ' '))
	return verse_pair


def edit_distance(x, y):
	# Levenshtein distance
	return editdistance.distance(x, y)


def edit_distance_ratio(x, y):
	# normalized Levenshtein distance ratio, larger is similar
	# we should remove "$" when perform comparison
	x = x.replace('$', '')
	y = y.replace('$', '')
	return ratio(x, y)

def create_concept_dir(store_dir, src_string):

	# store the results to a given directory
	if os.path.exists(store_dir):
		pass
	else:
		os.makedirs(store_dir)

	concept = '-'.join(src_string) if isinstance(src_string, list) else src_string
	if not os.path.exists(store_dir + '/' + concept):
		os.makedirs(store_dir + '/' + concept)
	return store_dir + '/' + concept


# sanity check (may be removed for efficiency)
def propose_new_candidates_by_edit_distance(full_indeces, aligned_indeces, concept_indeces, ignore_indeces, 
											parallel_verse_IDs, sentences, proposed_translations, results_list, check_num, last_result, results_to_store):
	"""
	concept_indeces: the rest indeces not macthed yet
	"""
	if len(proposed_translations) >= 5:
		return proposed_translations, check_num, last_result, results_to_store

	for candidate in results_list:
		# # print(candidate[0])
		for translation in proposed_translations:
			# # print(candidate[0], translation, edit_distance_ratio(candidate[0], translation))
			# # print(romanize([candidate[0]]), romanize([translation]), edit_distance_ratio(candidate[0], translation))
			if edit_distance_ratio(candidate[0], translation) > 0.7 and candidate[0] not in translation:
				# if the string is similar to a proposed translation
				# # print(candidate[0], translation, edit_distance_ratio(candidate[0], translation))
				result = compute_statistics([candidate[0]], sentences, concept_indeces, parallel_verse_IDs, ignore_indeces)
				if len(result) == 0:
					break # because the candidate can be out of concept_indeces
				# # print(result[candidate[]])
				new_parallel_verse_indx = [i for i in range(len(parallel_verse_IDs)) if parallel_verse_IDs[i] in result[candidate[0]][3][0]] # where it appears
				if len(new_parallel_verse_indx) == 0:
					break
				# # print(candidate[0], result[candidate[0]][2])
				aligned_indeces = sorted(list(set(aligned_indeces).union(set(new_parallel_verse_indx))))
				concept_indeces = sorted(list(set(full_indeces).difference(set(aligned_indeces))))
				ignore_indeces = sorted(list(set(full_indeces).intersection(set(aligned_indeces))))
				# # print(candidate[0], result[candidate[0]][2])
				# the string is not in the current rest unaligned verses
				last_result = (candidate[0], result[candidate[0]])
				results_to_store.append(last_result)
				check_num.append(len(new_parallel_verse_indx))
				proposed_translations = add_new_candidate(proposed_translations, candidate[0])
				# # print(proposed_translations)
				if len(concept_indeces) == 0 or len(proposed_translations) >= 5:
					return proposed_translations, check_num, last_result, results_to_store

	return proposed_translations, check_num, last_result, results_to_store


def add_new_candidate(proposed_candidates, string):
	if len(proposed_candidates) == 0:
		return [string]
	if string in proposed_candidates:
		return proposed_candidates
	else:
		proposed_candidates.append(string)
		return proposed_candidates


# to search the shortest word combination that has the search_sub_string
def search(search_sub_string, sent):

	temp = search_sub_string[1:] if search_sub_string[0] == '$' else search_sub_string
	temp = temp[:-1] if temp[-1] == '$' else temp
	search_len = len([char for char in temp if char == '$']) + 1
	search_subwords = temp.split('$')

	# add "$" to the first sub string or the last sub string if there are "$" in the original string
	if search_sub_string[0] == '$':
		search_subwords[0] = '$' + search_subwords[0]
	if search_sub_string[-1] == '$':
		search_subwords[-1] = search_subwords[-1] + '$'

	# add '$' to the end of the sub_string
	for i in range(len(search_subwords)):
		if i == len(search_subwords) - 1:
			break
		search_subwords[i] = search_subwords[i] + '$'

	# if the sent is already been replaced, comment the following line
	sent = '$' + sent.replace(' ', '$') + '$'
	words = sent.lower().split('$')
	words = ['$'+ word +'$' for word in words]
	for i in range(len(words)):
		if i >= len(words) - search_len + 1:
			break
		if search_subwords[0] in words[i]:
			flag = True
			for j in range(1, search_len):
				if search_subwords[j] in words[i+j]:
					pass
				else:
					flag = False
					break
			if flag:
				matched_expanded_string = words[i]
				for indx in range(i+1, i+search_len):
					matched_expanded_string += words[indx][1:] # remove the first '$'
				# matched_expanded_string = ''.join([words[indx] for indx in range(i, i+search_len)])
				return matched_expanded_string
			else:
				continue
	return ValueError


def expand_string(sub_string, statistics, src_sentences, tgt_sentences, parallel_verse_IDs):
	# result: a tuple which stores the information
	# result[1][3][0] #TF verse IDs
	verses = find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, statistics[sub_string][3][0])
	strings_candidates = set()
	for sent_pair in verses:
		# # print(sub_string)
		# # print(sent_pair[1])
		matched = search(sub_string, sent_pair[1])
		if matched:
			strings_candidates.add(matched)
	return strings_candidates


def main(params):
	start = time.time()
	# generate parser / parse parameters
	src_lang = params.src_lang
	src_string = params.src_string
	minimum_ngram_len = params.minimum_ngram_len
	ignore_case = params.ignore_case
	target_num = params.target_num
	ngram_min_len = params.ngram_min_len
	ngram_max_len = params.ngram_max_len
	make_annotation = params.make_annotation

	if params.use_edition_most:
		find_version = find_version_most
	else:
		find_version = find_version_new

	#considering multiple search strings for the same concept, e.g., $bird and $fowl, using ||| to split
	src_string = src_string if len(src_string.split('|||')) == 1 else src_string.split('|||')
	final_results_to_store ={}
	string_dict = {}
	tgt_versions = []
	tgt_file_names = []
	src_version, src_file_name = find_version(src_lang)

	# use the information of PBC
	pbc = PBC_filenames()

	if params.tgt_langs == 'all':
		tgt_langs = pbc.get_langs()
	else:
		tgt_langs = params.tgt_langs.split('-')
	for lang in tgt_langs:
		tgt_file, tgt_file_name = find_version(lang)
		tgt_versions.append(tgt_file)
		tgt_file_names.append(tgt_file_name)


	new_store_dir = create_concept_dir(params.store_dir, src_string)
	store_name = 'concept.results'

	# find whether a given language pair has been stored / computed (TO DO)
	langs_computed = []
	if os.path.exists(new_store_dir + '/' + store_name + '.pickle'):
		with open(new_store_dir + '/' + store_name + '.pickle', 'rb') as handle:
			final_results_to_store = pickle.load(handle)
		for key, value in final_results_to_store.items():
			langs_computed.append(key)
		langs_computed = sorted(langs_computed)

	# # print(langs_computed)
	# find the plausible translation of the concepts
	for version, tgt_lang in zip(tgt_versions, tgt_langs):
		# check whether the alignment for that language pair has been computed before
		if src_lang + '-' + tgt_lang in langs_computed:
			print(src_lang + ' (' + get_full_language_name(src_lang) + ') (src) - ' + tgt_lang + ' (' + get_full_language_name(tgt_lang) + ') (tgt)' )
			continue
		current_store_results = {}
		src_sentences, tgt_sentences, parallel_verse_IDs = obtain_parallel_contents(src_version, version)
		assert len(src_sentences) == len(tgt_sentences)
		candidate_translations, concept_verse_indeces = find_candidates_and_verses_indeces(src_string, src_sentences, tgt_sentences, 
																						  minimum_ngram_len, ignore_case=ignore_case,
																						  ngram_min_len=ngram_min_len,
																						  ngram_max_len=ngram_max_len
																						  )

		# if there are no parallel verses that have the concepts, we skip this language.
		if len(concept_verse_indeces) == 0:
			continue

		# using multiprocessing here
		if params.multiprocessing:
			cores_num = multiprocessing.cpu_count()
			# # print(cores_num)
			results = []
			pool = multiprocessing.Pool(cores_num)
			candidate_chunked = np.array_split(candidate_translations, cores_num)
			final_dict = {}
			for i in range(cores_num):
				candidate_parts = candidate_chunked[i].tolist()
				results.append(pool.apply_async(compute_statistics, (candidate_parts, tgt_sentences, concept_verse_indeces, parallel_verse_IDs, [], False, True)))
			pool.close()
			pool.join()
			for r in results:
				final_dict.update(r.get())
			results = final_dict
		else:
			results = compute_statistics(candidate_translations, tgt_sentences, concept_verse_indeces, parallel_verse_IDs, [], is_reverse=False, is_first_search=True)

		# if there is no results obatined (empty dictionary)
		if len(results) == 0:
			continue

		# get the rest possible candidate translations
		# candidate_translations = [key for key, value in results.items()]
		results = clean_results(results, len(results))
		# # print([(result[0], result[1][0], result[1][2]) for result in results[:100]])
		results = clean_results(filter_string(dict(results)), len(results))
		# # print([(result[0], result[1][0], result[1][2]) for result in results[:100]])
		# get the rest possible candidate translations
		candidate_translations = [key for key, value in dict(results).items()]

		# print(len(results))
		# print(results)
		is_first_search = False
		# if the first search is even smaller than 10 times of the occurence, the concept might have multiple possible  
		if results[0][1][2][0][0] <= min(max(1, len(concept_verse_indeces) // 10), 10):
			is_first_search = True

		if len(results) == 0:
			continue


		if len(results) == 0:
			continue # if no matched we just skip this language

		# check 1-to-n problem:
		check_num = []
		check_num.append(results[0][1][2][0][0])
		one_to_n = check_num
		true_positive_verse_num = sum(check_num)
		translations = [results[0][0]]
		last_one_to_n_results = []
		iter_time = 0
		parallel_verse_indx = [i for i in range(len(parallel_verse_IDs)) if parallel_verse_IDs[i] in results[0][1][3][0]]

		compare_strings = [r[0] for r in results[:5]]
		# compare_strings = [results[i][0] for i in range(5)]
		# # print(compare_strings)
		# # print(results[0][0], ': ', results[0][1][2])
		one_to_n_flag = False
		last_new_results = []
		last_win_idx = 0

		# recursive search results to store
		one_to_n_results_to_store = []
		if sum(check_num) != len(concept_verse_indeces):
			one_to_n_flag = True

		one_to_n_coverage_ratio = sum(check_num) / len(concept_verse_indeces) # if this ratio reaches certain value, we can stop the reverse search
		# # print(one_to_n_coverage_ratio)
		while one_to_n_coverage_ratio < params.threshold and iter_time < 5: 
		#and iter_time <= 2:
			# # print(one_to_n_coverage_ratio)
			one_to_n_flag = True
			# remove the verses in common
			# # print(parallel_verse_IDs)
			# # print(results[0][1][3][0])
			# # print(parallel_verse_indx)
			current_concept_verse_indeces = sorted(list(set(concept_verse_indeces).difference(set(parallel_verse_indx))))
			ignore_conept_verse_indeces = sorted(list(set(concept_verse_indeces).intersection(set(parallel_verse_indx))))
			# using multiprocessing here
			if params.multiprocessing:
				cores_num = min(multiprocessing.cpu_count(), len(candidate_translations))
				new_results = []
				pool = multiprocessing.Pool(cores_num)
				candidate_chunked = np.array_split(candidate_translations, cores_num)
				final_dict = {}
				for i in range(cores_num):
					candidate_parts = candidate_chunked[i].tolist()
					new_results.append(pool.apply_async(compute_statistics, (candidate_parts, tgt_sentences, 
								   current_concept_verse_indeces, parallel_verse_IDs, ignore_conept_verse_indeces, False, is_first_search)))
				pool.close()
				pool.join()
				for r in new_results:
					final_dict.update(r.get())
				new_results = final_dict
			else:
				new_results = compute_statistics(candidate_translations, tgt_sentences, current_concept_verse_indeces, 
											 parallel_verse_IDs, ignore_conept_verse_indeces, is_reverse=False, is_first_search=is_first_search)

			# get the rest possible candidate translations
			candidate_translations = [key for key, value in new_results.items()]
			# # print(len(candidate_translations))

			new_results = clean_results(new_results, 100)
			# # print([(result[0], result[1][0], result[1][2]) for result in new_results[:10]])

			# when no gain or the gain is very small (e.g., 1), we stop
			if len(new_results) == 0 or new_results[0][1][2][0][0] <= 1:
				new_results = last_new_results.copy()
				win_idx = last_win_idx
				break

			new_results = clean_results(filter_string(dict(new_results)), 5)
			# # print([(result[0], result[1][0], result[1][2]) for result in new_results[:10]])

			# we have to deal with the problem that there might be multiple strings with the same statistics (chisquare score)
			candidates = [0]
			for i in range(1, len(new_results)):
				if new_results[i][1][0] == new_results[0][1][0]:
					candidates.append(i)
				else:
					break
			if len(candidates) == 1:
				win_idx = candidates[0]
			else:
				distances = [0 for i in range(len(candidates))]
				for i in range(len(candidates)):
					for compare_string in compare_strings:
						distances[i] += edit_distance_ratio(new_results[candidates[i]][0], compare_string)
				win_idx = candidates[distances.index(max(distances))]

			# # print(new_results[win_idx][0], new_results[win_idx][1][0], new_results[win_idx][1][2])

			new_parallel_verse_indx = [i for i in range(len(parallel_verse_IDs)) if parallel_verse_IDs[i] in new_results[win_idx][1][3][0]]
			parallel_verse_indx = sorted(list(set(parallel_verse_indx).union(set(new_parallel_verse_indx))))

			translations = add_new_candidate(translations, new_results[win_idx][0])

			check_num.append(new_results[win_idx][1][2][0][0])

			one_to_n_results_to_store.append(new_results[win_idx])

			one_to_n_coverage_ratio = sum(check_num) / len(concept_verse_indeces)

			last_new_results = new_results.copy()
			last_win_idx = win_idx
			one_to_n = check_num
			# # print(one_to_n)
			# # print(len(candidate_translations))
			iter_time += 1

			# if no further increase, we should end the loop
			if true_positive_verse_num == sum(check_num):
				break
			else:
				true_positive_verse_num = sum(check_num)

		if one_to_n_flag and iter_time >= 1:
			last_one_to_n_results = new_results[win_idx]

		# when there are still verses not matched and the loop is not performed, we still want to perform a sanity check, then
		if one_to_n_flag and iter_time == 0:
			new_results = results
			last_one_to_n_results = new_results[0]

		# # if all the false negatives are moved
		# if one_to_n_flag and sum(check_num) == len(concept_verse_indeces):
		# 	last_one_to_n_results = []

		# maybe we do not need to do it for efficiency
		# # # print(one_to_n_coverage_ratio)
		# # we only do this sanity check if at least one search loop is performed, and there are still verses not matched
		# if one_to_n_flag and sum(check_num) != len(concept_verse_indeces):
		# 	current_concept_verse_indeces = sorted(list(set(concept_verse_indeces).difference(set(parallel_verse_indx))))
		# 	ignore_conept_verse_indeces = sorted(list(set(concept_verse_indeces).intersection(set(parallel_verse_indx))))
		# 	# sanity check
		# 	translations, check_num, last_one_to_n_results, one_to_n_results_to_store = propose_new_candidates_by_edit_distance(concept_verse_indeces, parallel_verse_indx, 
		# 											current_concept_verse_indeces, ignore_conept_verse_indeces, parallel_verse_IDs, tgt_sentences, 
		# 											translations, new_results, check_num, last_one_to_n_results, one_to_n_results_to_store)
		# 	one_to_n = check_num

		# time_one_to_n = time.time()
		# print('Run time til 1-to-n: ',time_one_to_n - start)

		# check n-to-1 problem

		# n_to_one now should be the sum of all the false positives
		n_to_one = [results[0][1][2][0][1]]
		for result in one_to_n_results_to_store:
			n_to_one.append(result[1][2][0][1])

		n_to_one = sum(n_to_one)
		possible_tgt_string_translations = []

		# if the most associated target string has only 1 as OVERREACH 
		n_to_one_flag = False
		# the possible translation from the most associated target string to source languages
		last_n_to_one_results = []

		# the reverse results to store
		n_to_one_results_to_store = []

		if n_to_one != 0:
			n_to_one_flag = True
			# doing reverse search
			# search_string = results[0][0]
			search_string = translations[:]

			# first use the src string as the candidate (for english, we want prevent search_string length less than 3)
			# (for english, we want also window_size=1)
			reverse_candidate_translations, reverse_concept_verse_indeces = find_candidates_and_verses_indeces(search_string, tgt_sentences, 
																			src_sentences, minimum_ngram_len=3, ignore_case=ignore_case, 
																			ngram_min_len=3, ngram_max_len=ngram_max_len, find_word_ngram=True)

			# check if there are multiple possible source translations
			check_num = [0]
			iter_time = 0
			parallel_verse_indx = []
			true_positive_verse_num = sum(check_num)
			src_string_len = 1 if isinstance(src_string, str) else len(src_string)
			src_string_for_search = [src_string] if isinstance(src_string, str) else src_string[:]
			last_new_results = []
			last_win_idx = 0
			current_concept_verse_indeces = reverse_concept_verse_indeces[:]

			n_to_one_coverage_ratio = sum(check_num) / len(reverse_concept_verse_indeces) # if this ratio reaches certain value, we can stop the reverse search
			compare_strings = [src_string] if isinstance(src_string, str) else src_string[:]
			while (n_to_one_coverage_ratio < params.threshold and iter_time < 5 + src_string_len) or iter_time < src_string_len: 
			#and iter_time <= 2 + src_string_len:
				# # print(n_to_one_coverage_ratio)
				current_concept_verse_indeces = sorted(list(set(reverse_concept_verse_indeces).difference(set(parallel_verse_indx))))
				ignore_conept_verse_indeces = sorted(list(set(reverse_concept_verse_indeces).intersection(set(parallel_verse_indx))))
				# using multiprocessing here
				if params.multiprocessing and iter_time >= src_string_len:
					cores_num = min(multiprocessing.cpu_count(), len(reverse_candidate_translations))
					new_results = []
					pool = multiprocessing.Pool(cores_num)
					candidate_chunked = np.array_split(reverse_candidate_translations, cores_num)
					final_dict = {}
					for i in range(cores_num):
						candidate_parts = candidate_chunked[i].tolist()
						new_results.append(pool.apply_async(compute_statistics, (candidate_parts, src_sentences, 
									   current_concept_verse_indeces, parallel_verse_IDs, ignore_conept_verse_indeces, True)))
					pool.close()
					pool.join()
					for r in new_results:
						final_dict.update(r.get())
					new_results = final_dict
				elif iter_time < src_string_len:
					# search the src_string first
					new_results = compute_statistics(src_string_for_search, src_sentences, reverse_concept_verse_indeces, parallel_verse_IDs, ignore_conept_verse_indeces, is_reverse=True)
				else:
					new_results = compute_statistics(reverse_candidate_translations, src_sentences, current_concept_verse_indeces, 
												 parallel_verse_IDs, ignore_conept_verse_indeces, is_reverse=True)

				# we should also filter out the candidates in reverse_candidate_translations which has 0 in n00
				if iter_time >= src_string_len:
					if iter_time == src_string:
						# decrease the candidates at the first step
						temp_results = clean_results(new_results, len(new_results))
						temp_results = clean_results(filter_string(dict(temp_results), minimum_len=4), len(temp_results))
						reverse_candidate_translations = [key for key, value in temp_results.items()]
					else:
						reverse_candidate_translations = [key for key, value in new_results.items()]

				new_results = clean_results(new_results, 100)

				# if the results are empty, we either stop the search (no further gain), or continue search (when we are still searching in src_string_for_search)
				if len(new_results) == 0 or (new_results[0][1][2][0][0] <= 1 and iter_time >= src_string_len):
					new_results = last_new_results.copy()
					win_idx = last_win_idx
					if iter_time < src_string_len:
						iter_time += 1
						continue
					else:
						break

				# for the reverse search in English, we would like to force the resulted string to be longer than 3 (we already done that)
				new_results = clean_results(filter_string(dict(new_results)), 5)
				# # print(n_to_one, len(reverse_concept_verse_indeces))
				# # print(sum(check_num))
				# # print([(result[0], result[1][0], result[1][2]) for result in new_results[:10]])

				# we have to deal with the problem that there might be multiple strings with the same statistics
				# finding candaites with the minimum edit distance with the already selected strings
				candidates = [0]
				for i in range(1, len(new_results)):
					if new_results[i][1][0] == new_results[0][1][0]:
						candidates.append(i)
					else:
						break
				if len(candidates) == 1:
					win_idx = candidates[0]
				else:
					distances = [0 for i in range(len(candidates))]
					for i in range(len(candidates)):
						for compare_string in compare_strings:
							distances[i] += edit_distance_ratio(new_results[candidates[i]][0], compare_string) / len(compare_strings) # average ratio
					win_idx = candidates[distances.index(max(distances))]

				new_parallel_verse_indx = [i for i in range(len(parallel_verse_IDs)) if parallel_verse_IDs[i] in new_results[win_idx][1][3][0]]
				parallel_verse_indx = sorted(list(set(parallel_verse_indx).union(set(new_parallel_verse_indx))))

				if iter_time < src_string_len:
					src_string_for_search.remove(new_results[win_idx][0])

				possible_tgt_string_translations = add_new_candidate(possible_tgt_string_translations, new_results[win_idx][0])

				if new_results[win_idx][0] not in compare_strings:
					compare_strings.append(new_results[win_idx][0])

				check_num.append(new_results[win_idx][1][2][0][0])
				iter_time += 1
				last_new_results = new_results.copy()
				last_win_idx = win_idx
				n_to_one_results_to_store.append(new_results[win_idx])

				n_to_one_coverage_ratio = sum(check_num) / len(reverse_concept_verse_indeces)
				# if no further increase, we should end the loop
				if true_positive_verse_num == sum(check_num):
					break
				else:
					true_positive_verse_num = sum(check_num)

			# # if all the false positives are filtered
			# if sum(check_num) == len(reverse_concept_verse_indeces):
			# 	last_n_to_one_results = []


			# the loop will at least be performed for one time (src string)
			if iter_time >= 1:
				last_n_to_one_results = new_results[win_idx]

			# # we only do this sanity check if at least one search loop is performed, and there are still verses not matched
			# if iter_time >= 1 and sum(check_num) != len(reverse_concept_verse_indeces):
			# 	last_n_to_one_results = new_results[win_idx]
			# 	# if the current coverage exceeds the threshold, the statistics can be less-trustible, we move to use edit distance to find candidates if there are any
			# 	current_concept_verse_indeces = sorted(list(set(reverse_concept_verse_indeces).difference(set(parallel_verse_indx))))
			# 	ignore_conept_verse_indeces = sorted(list(set(reverse_concept_verse_indeces).intersection(set(parallel_verse_indx))))
			# 	possible_tgt_string_translations, check_num, last_n_to_one_results, n_to_one_results_to_store = propose_new_candidates_by_edit_distance(reverse_concept_verse_indeces, 
			# 											parallel_verse_indx, current_concept_verse_indeces, ignore_conept_verse_indeces, parallel_verse_IDs, 
			# 											src_sentences, possible_tgt_string_translations, new_results, check_num, last_n_to_one_results, n_to_one_results_to_store)
			# # else:
			# # 	n_to_one_flag = False # if only reverse search only does one time and the result is not significant, then we restore false for the flag (only one word found).

		# time_n_to_one = time.time()
		# print('Run time til n-to-1: ',time_n_to_one - start)


		# things needed to be written:
		if make_annotation:
			new_print_dir = create_concept_dir(params.print_dir, src_string)
			if isinstance(src_string, str):
				f = open(new_print_dir + "/" + src_lang + '-' + tgt_lang + '-' + src_string + '.md', "w", encoding="utf-8")
			else:
				f = open(new_print_dir + "/" + src_lang + '-' + tgt_lang + '-' + '-'.join(src_string) + '.md', "w", encoding="utf-8")

		print(src_lang + ' (' + get_full_language_name(src_lang) + ') (src) - ' + tgt_lang + ' (' + get_full_language_name(tgt_lang) + ') (tgt)' )
		# print("--------------------------------------------")
		# print(src_lang + ': ' + src_version.split('/')[-1])
		# print(tgt_lang + ': ' + version.split('/')[-1])
		# print()

		if make_annotation:
			f.write("**" + src_lang + ' (' + get_full_language_name(src_lang) + ') (src) - ' + tgt_lang + ' (' + get_full_language_name(tgt_lang) + ') (tgt)**' + '  \n')
			f.write("--------------------------------------------" + '  \n')
			f.write('  \n')
			f.write(src_lang + ': ' + src_version.split('/')[-1] + '  \n')
			f.write(tgt_lang + ': ' + version.split('/')[-1]+ '  \n')
			f.write('  \n')

		# things to be written
		current_store_results['src_version'] = src_version.split('/')[-1]
		current_store_results['tgt_version'] = version.split('/')[-1]
		current_store_results['candidate_translations'] = translations # the translations of the source concept (1 to n)
		current_store_results['tgt_string_translations'] = possible_tgt_string_translations # the possible translations of the most associated target string to src lang (n to 1)
		current_store_results['parallel_verse_IDs'] = parallel_verse_IDs
		current_store_results['statistics1'] = results[0] # to store the statistics of the most associated target string
		current_store_results['statistics2'] = n_to_one_results_to_store if n_to_one_flag else []
		current_store_results['statistics3'] = one_to_n_results_to_store if one_to_n_flag else []

		# print('Source search string: ', src_string if isinstance(src_string, str) else str(src_string)) 
		if make_annotation:
			write_str = src_string if isinstance(src_string, str) else str(src_string)
			f.write('Source recursive search string: ' + write_str + '  \n')
		# print('\#Verses containing the source search string: ', len(concept_verse_indeces), 
			  # ' - \#Verses accumulatively finally matched: ', sum(one_to_n))
		if make_annotation:
			f.write('\#Verses containing the source search string: ' + str(len(concept_verse_indeces)) + 
					' - \#Verses accumulatively finally matched: ' + str(sum(one_to_n)) + '  \n')

		if make_annotation:
			# a flag for whether romanized version should be included
			need_romanized = requires_romanize(version.split('/')[-1])
			if not need_romanized:
				pass
			else:
				romanized_translations = romanize(translations, lang=tgt_lang)

			# the table needed to be printed out
			statistics = [current_store_results['statistics1']]
			statistics += current_store_results['statistics3']
			statistics = dict(statistics)

			# print()
			f.write('  \n')

			table_srr = "| \|-strings-\| | \|-chisquare-\| | \|-#TP-\| | \|-#FP-\| |" + "\n"
			table_srr += "| :----------| :----------:  | :----------: | :----------: |" + "\n"
			for indx in range(len(translations)):
				# if indx == 0:
				current_str = '| '
				current_str += '\|' + translations[indx]
				current_str +=  ('( ' + romanized_translations[indx] + ') ' if need_romanized else '') + '\|'
				current_str += ' | \|' + str(int(statistics[translations[indx]][0])) + '\|'
				current_str += (' | \|' + str(statistics[translations[indx]][2][0][0]) + '\| | \|' +  str(statistics[translations[indx]][2][0][1]))
				current_str += '\| |\n'
				table_srr += current_str

			# print('Table showing the statistics of the found target strings:  \n')
			f.write('Table showing the statistics of the found target strings:  \n')

			# print()
			f.write('  \n')

			# print(table_srr)
			f.write(table_srr)

			# print()
			f.write('  \n')

			# showing the statistics of the reverse search

			# if multiple tgt string translations availale
			if len(possible_tgt_string_translations) > 1 or n_to_one_flag:
				# # print("The most associated target string ", translations[0],  " has the following associated strings in the source language: ", possible_tgt_string_translations)
				# f.write("The most associated target string " + translations[0] + " has the following associated strings in the source language: " 
				# 		+  str(possible_tgt_string_translations) + '  \n')
				statistics2 = dict(current_store_results['statistics2'])

				table_srr = "| \|-strings-\| | \|-#TP-\| | \|-#FP-\| | " + "\n"
				table_srr += "| :---------- | :----------: | :----------: | " + "\n"
				total_numer = 0
				for indx in range(len(possible_tgt_string_translations)):
					current_str = '| \|'
					current_str += possible_tgt_string_translations[indx]
					current_str += ('\| | \|' + str(statistics2[possible_tgt_string_translations[indx]][2][0][0]) + '\| | \|' +\
									  str(statistics2[possible_tgt_string_translations[indx]][2][0][1]) + '\| | \n') 
					total_numer += statistics2[possible_tgt_string_translations[indx]][2][0][0]
					table_srr += current_str
				table_srr += "| \|Not matched\|" + " | \|" + str(len(reverse_concept_verse_indeces)-total_numer) + "\| | " + " | \n "
				table_srr += "| \|total number of occurrences\|" + "| \|" + str(len(reverse_concept_verse_indeces)) + "\| | " + " | \n "


				# print('Target reverse search string: ', translations if isinstance(translations, str) else str(translations))
				write_str = translations if isinstance(translations, str) else str(translations)
				f.write('Target reverse search string: ' + write_str + '  \n')
				# print('\#Verses containing the target search string: ', len(reverse_concept_verse_indeces), 
				  	# ' - \#Verses accumulatively finally matched: ', total_numer)
				f.write('\#Verses containing the target search string: ' + str(len(reverse_concept_verse_indeces)) + 
					' - \#Verses accumulatively finally matched: ' + str(total_numer) + '  \n')

				# print()
				f.write('  \n')

				# print('Table showing the statistics of the reverse search:  \n')
				f.write('Table showing the statistics of the reverse search:  \n')

				# print()
				f.write('  \n')

				# print(table_srr)
				f.write(table_srr)

				# print()
				f.write('  \n')

			# print()
			f.write('  \n')

			# print("--------------------------------------------")
			f.write("--------------------------------------------" + '  \n')

			# print("True Positive Sample Verses:")
			f.write("**True Positive Sample Verses:**" + '  \n')

			# print()
			f.write('  \n')

			i = 0
			for tr in translations:
				# # print(expand_string(tr, statistics, src_sentences, tgt_sentences, parallel_verse_IDs))
				# print("**" + tr + '**: ')
				# print(f"Expanded strings: {str(dict(statistics[tr][4]))}") 

				f.write("**" + tr + '**: ' + '  \n')
				f.write(f"Expanded strings: {str(dict(statistics[tr][4]))}  \n")
				f.write('  \n')
				temp = find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, statistics[tr][3][0][:target_num])
				for t in temp:
					# # print(src_string)
					# # print(translations)
					# # print(t)
					highlighted_pair = highlight_string(t, src_string, translations)
					# print(highlighted_pair)
					f.write("**" + src_lang + '**: ' + highlighted_pair[0] + '  \n')
					f.write("**" + tgt_lang + '**: ' + highlighted_pair[1] + '  \n')
					if need_romanized:
						# print('Romanized verse pair: ')
						highlighted_pair = highlight_string((t[0], romanize([t[1]], lang=tgt_lang)[0]), src_string, romanized_translations)
						# print(highlighted_pair)
						f.write('**romanized**: ' + highlighted_pair[1] + '  \n')

					# print()
					f.write('  \n')
				i += 1

			# print()
			f.write('  \n')

			# print()
			f.write('  \n')
			current_store_results['tp_samples'] = temp

			# print("--------------------------------------------")
			f.write("--------------------------------------------" + '  \n')

			# print("False Positive Sample Verses:")
			f.write("**False Positive Sample Verses:**" + '  \n')

			# print()
			f.write('  \n')

			i = 0
			for tr in translations:
				# print("**" + tr + '**: ')
				f.write("**" + tr + '**: ' + '  \n')

				# print(f"Expanded strings: {str(dict(statistics[tr][5]))}") 
				f.write(f"Expanded strings: {str(dict(statistics[tr][5]))}  \n")
				f.write('  \n')
				temp = find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, statistics[tr][3][1][:target_num])
				for t in temp:
					highlighted_pair = highlight_string(t, src_string, translations)
					# print(highlighted_pair)
					f.write("**" + src_lang + '**: ' + highlighted_pair[0] + '  \n')
					f.write("**" + tgt_lang + '**: ' + highlighted_pair[1] + '  \n')
					if need_romanized:
						# print('Romanized verse pair: ')
						highlighted_pair = highlight_string((t[0], romanize([t[1]], lang=tgt_lang)[0]), src_string, romanized_translations)
						# print(highlighted_pair)
						f.write('**romanized**: ' + highlighted_pair[1] + '  \n')

					# print()
					f.write('  \n')
				i += 1

				# print()
				f.write('  \n')

			# print()
			f.write('  \n')

			# print()
			f.write('  \n')

			current_store_results['fp_samples'] = temp

			# print("--------------------------------------------")
			f.write("--------------------------------------------" + '  \n')

			# print("False Negative Sample Verses:")
			f.write("**False Negative Sample Verses:**" + '  \n')

			# when one_to_n_flag always use last_one_to_n_results to reduce inconsistency
			if one_to_n_flag: # if more words are found
				temp = find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, last_one_to_n_results[1][3][2][:5])
			else:
				temp = find_verse(src_sentences, tgt_sentences, parallel_verse_IDs, results[0][1][3][2][:5])
			for t in temp:
				highlighted_pair = highlight_string(t, src_string, translations)
				# print(highlighted_pair)
				f.write("**" + src_lang + '**: ' + highlighted_pair[0] + '  \n')
				f.write("**" + tgt_lang + '**: ' + highlighted_pair[1] + '  \n')
				if need_romanized:
					# print('Romanized verse pair: ')
					highlighted_pair = highlight_string((t[0], romanize([t[1]], lang=tgt_lang)[0]), src_string, romanized_translations)
					# print(highlighted_pair)
					f.write('**romanized**: ' + highlighted_pair[1] + '  \n')

				# print()
				f.write('  \n')

			# print()
			f.write('  \n')
			current_store_results['fn_samples'] = temp

			f.close()

		final_results_to_store[src_lang + '-' + tgt_lang] = current_store_results

	with open(new_store_dir + '/' + store_name + '.pickle', 'wb') as handle:
		pickle.dump(final_results_to_store, handle)
	# print(final_results_to_store)
	end = time.time()
	print('Run time: ',end - start)

if __name__ == '__main__':
	# generate parser / parse parameters
	parser = get_parser()
	params = parser.parse_args()
	main(params)
