#!/usr/bin/env python3
import codecs
import argparse
import os.path
import os

class Uroman:
	def __init__(self):
		pass

	def romanize(self, sentences, temp_path='/mounts/Users/student/yihong/Documents/concept_align/temp', lang=None):
		# sentences: a list of sentences
		uroman_path = "./uroman/bin/"

		# create parallel text
		in_path = temp_path + '/sentence' + ".txt"
		out_path = temp_path + '/sentence_roman' + ".txt"

		fa_file = codecs.open(in_path, "w", "utf-8")
		for sentence in sentences:
			fa_file.write(sentence + "\n")
		fa_file.close()

		if lang is None:
			os.system(uroman_path + "uroman.pl < {0} > {1} ".format(in_path, out_path))
		else:
			os.system(uroman_path + "uroman.pl -l {0} < {1} > {2} ".format(lang, in_path, out_path))

		romanize_sentences = []
		f1 = open(out_path, "r", encoding='utf-8')
		for line in f1.readlines():
			romanize_sentences.append(line.strip())

		os.system("rm {}".format(in_path))
		os.system("rm {}".format(out_path))

		return romanize_sentences

# sentences1 = ['今天真是一个好天气。', '哈哈哈我爱刘一宏。']
# sentences2 = ['Ich habe Vögel gesehen!', '$This is a test of ^ ! #']
# sentences3 = ['أرى طائرًا يطير في السماء', 'انا الاله الوحيد']

# uroman = Uroman()
# print(uroman.romanize(sentences1, './temp'))
# print(uroman.romanize(sentences2, './temp', lang='deu'))
# print(uroman.romanize(sentences3, './temp', lang='dfa'))