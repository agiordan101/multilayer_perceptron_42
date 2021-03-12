import numpy as np
# import pandas as pd

class DataProcessing():

	def __init__(self, data, columns=None):

		# print(f"Data: {data}")
		# if not isinstance(data, pd.DataFrame):
		# 	data = pd.DataFrame(data=data, columns=columns)
		self.df = data
		self.columns = columns
		self.normalization_data = []
		self.standardization_data = []

	def normalize(self):

		# new_lst = []
		data = {}

		if self.normalization_data:
			
			for item, data in zip(self.df.items(), self.normalization_data):
				# new_lst.append([(x - data[0]) / (data[1] - data[0]) for x in item[1].values])
				data[item[0]] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column.values]

		else:
			for feature, column in self.df.items():
				_min = min(column)
				_max = max(column)
				# _min = column.min()
				# _max = column.max()
				self.normalization_data.append([_min, _max])
				data[feature] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column]
				# data[feature] = [(x - _min) / (_max - _min) for x in column.values]

		self.df = data
		# self.df = pd.DataFrame(data=data, columns=self.columns)

	def get_data(self, data_type="2d_np_array"):
		
		return np.array([np.array(features) for features in zip(*list(self.df.values()))])

	def save_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'w') as f:

			if normalization:
				f.write("Normalization data\n")
				for _min, _max in self.normalization_data:
					f.write(f"{_min}/{_max}\n")
			
			if standardization:
				f.write("Standardization data\n")
				for mean, std in self.standardization_data:
					f.write(f"{mean}/{std}\n")

			f.close()

	def load_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'r') as f:
			data = f.read()
			f.close()

			if normalization:
				self.normalization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"normalization_data: {self.normalization_data}")

			if standardization:
				self.standardization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"standardization_data: {self.standardization_data}")
