#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


class PrepareData(object):

	def __init__(self, filename="sentiment_analysis_trainingset.csv"):
		data = pd.read_csv('data/' + filename, encoding = "utf-8")
		self.data =  data[['content','others_overall_experience']]

	def get_data(self):
		data = self.data[self.data.others_overall_experience != 0]
		for idx,row in data.iterrows():
			row[0] = row[0].replace(u'～', ' ').replace(u'，', ' ').replace(u'！', ' ')
		label = ['p' if x>0 else 'n' for x in data.others_overall_experience.values]
		data['label'] = label
		return data

if __name__ == '__main__':
	ppd = PrepareData()
	data = ppd.get_data()
