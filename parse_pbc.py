import numpy as np
import random
import pandas as pd
import regex


class PBC_filenames:
	"""docstring for ClassName"""
	def __init__(self, csv_path='/mounts/Users/student/yihong/Documents/concept_align'):
		self.csv_path = csv_path
		self.pbc_info = pd.read_csv('pbc_table.csv', converters={"language_code": str})

	def get_langs(self):
		langs = self.pbc_info['language_code'].values
		langs = sorted(list(set(langs)))
		return langs

	def get_file_name(self, lang, script_code=None):
		# results are a list of filenames
		results = self.pbc_info.loc[self.pbc_info['language_code']==lang]['file_name'].values
		if len(results) == 0:
			raise ValueError
		if script_code == None:
			return results
		else:
			filtered_results =[]
			for i in range(len(results)):
				 code = self.pbc_info.loc[self.pbc_info['language_code']==results[i]]['script_code'].values[0]
				 if code == script_code:
				 	filtered_results.append(results[i])
			return filtered_results