#!/usr/bin/env python3
import codecs
import os
import time
import argparse


def get_parser():
	"""
	Generate a parameters parser.
	"""
	parser = argparse.ArgumentParser(description="concept alignment")

	# main parameters
	parser.add_argument("--src_lang", type=str, help="the source language for the concepts to be computed")
	parser.add_argument("--concept_path", type=str, help="the path of the selected concepts file")
	parser.add_argument("--tgt_langs", type=str, help="the languguages considered", default='none')
	return parser

"""
export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/swadesh_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_swadesh.txt 2>&1 &

export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/bible_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_bible.txt 2>&1 &

export PYTHONIOENCODING=utf8; nohup python -u ./concept_script.py --src_lang eng --concept_path "./concepts/all_concepts.txt" --tgt_langs "all" > ./temp/log_run_all_bible.txt 2>&1 &

"""

class ConceptAligner:
	def __init__(self, src_lang, concept_path):
		self.src_lang = src_lang
		self.concept_list = []
		with open(concept_path, 'r', encoding="utf-8") as f:
			for line in f.readlines():
				self.concept_list.append(line.strip())

	def run_concepts_alignments(self, tgt_langs=None):
		if tgt_langs is None:
			tgt_langs = 'all'
		for concept in self.concept_list:
			print("Computing the concept '{0}'...".format(concept))
			command = "export PYTHONIOENCODING=utf8; python -u ./conceptualizer.py --src_lang '{0}' --src_string '{1}'  --tgt_langs '{2}' --target_num 2 --minimum_ngram_len 1 --use_edition_most true  --threshold 0.9 > ./temp/log_{3}.txt".format(self.src_lang, concept, tgt_langs, concept.replace('$', '\$').replace('|||', '_'))
			print(command)
			os.system(command)
			print()


def main(params):
	concept_aligner = ConceptAligner(params.src_lang, params.concept_path)
	if params.tgt_langs == 'none':
		tgt_langs = None
	else:
		tgt_langs = params.tgt_langs
	concept_aligner.run_concepts_alignments(tgt_langs)


if __name__ == '__main__':
	start = time.time()
	# generate parser / parse parameters
	parser = get_parser()
	params = parser.parse_args()
	main(params)
	end = time.time()
	print('Run time: ',end - start)