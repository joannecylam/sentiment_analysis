#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


class PrepareData(object):

	def __init__(self, filename="sentiment_analysis_trainingset.csv"):
		data = pd.read_csv('data/' + filename, encoding = "utf-8")
		self.data =  data[['content','others_overall_experience']]

	def get_data(self):
		# remove neutral label
		data = self.data
		print "before filter: ", self.data.shape
		data = data[data.others_overall_experience != 0]
		print "filter neutral: ", data.shape
		# remove null label
		data = data[data.others_overall_experience != -2]
		print "filter null: ", data.shape
		for idx,row in data.iterrows():
			row[0] = row[0].replace(u'～', ' ').replace(u'，', ' ').replace(u'！', ' ')
		label = ['p' if x>0 else 'n' for x in data.others_overall_experience.values]
		data['label'] = label
		return data

if __name__ == '__main__':
	ppd = PrepareData()
	data = ppd.get_data()
