import numpy as np

class DataProcessing():

	def __init__(self,
					features=None,
					targets=None,
					dataset_path=None,
					columns=None):

		if features and targets:

			if not isinstance(features, dict):
				raise Exception("Data parameter is not a instance Dict")
		# else:

			# if not dataset_path:
			# 	raise Exception("data or dataset_path arguments must be past")
			# self.parse_dataset(dataset_path)

		# print(f"features:\n{self.features}")
		# print(f"targets:\n{self.targets}")

		self.features = features
		self.targets = targets
		self.columns = columns
		self.normalization_data = []
		self.standardization_data = []

	def parse_dataset(self,
						dataset_path,
						columns_name=[],
						columns_range=[0, None],
						rows_range=[0, -1],
						parse_targets=True,
						target_index=1):

		try:
			#Open dataset file
			try:
				dataset_file = open(dataset_path, 'r')
				features_str = dataset_file.read()
				dataset_file.close()
			except Exception as error:
				print(f"[DataProcessing ERROR] Can't open file:\n{error}")
				exit(0)

			# Init data structure
			targets = []
			features = {}
			if columns_name:
				for feature in columns_name:
					features[feature] = []

			# Select rows
			features_str_split = features_str.split("\n")
			if rows_range[0] < 0 or len(features_str_split) <= rows_range[1]:
				print(f"[DataProcessing ERROR] rows_range parameter in parse_dataset() can't match with rows count:\nNumber of rows: {len(features_str_split)}\nrows_range: {rows_range}\n")
				rows_range = [0, -1]
			features_str_split = features_str_split[rows_range[0]:rows_range[1]] if rows_range[1] else features_str_split[rows_range[0]:]

			n_features = -1
			# Fill
			for r, student_str in enumerate(features_str_split):

				# Select features
				student_strlst = student_str.split(',')
				if columns_range[0] < 0 or (columns_range[1] and len(student_strlst) <= columns_range[1]):
					print(f"[DataProcessing ERROR] columns_range parameter in parse_dataset() can't match with features count:\nNumber of features: {len(student_strlst)}\ncolumns_range: {columns_range}\n")
					columns_range = [0, -1]
				student_strlst = student_strlst[columns_range[0]:columns_range[1]] if columns_range[1] else student_strlst[columns_range[0]:]

				if n_features == -1:
					n_features = len(student_strlst)
				elif n_features != len(student_strlst):
					print(f"Same numbers of features is required, row {r} has {len(student_strlst)} features instead of {n_features}")
					exit(0)

				if not features:
					for i in range(len(student_strlst) - (1 if parse_targets else 0)):
						features[f"feature_{i}"] = []

				if parse_targets:
					targets.append(student_strlst[target_index])
					student_strlst.pop(target_index)

				if columns_name:
					for i, feature in enumerate(columns_name):
						features[feature].append(float(student_strlst[i]) if student_strlst[i] else 0)
				else:
					for i, data in enumerate(student_strlst):
						features[f"feature_{i}"].append(float(data) if data else 0)

			self.features = features
			self.targets = targets

		except Exception as error:
			print(f"[DataProcessing ERROR] Error while parse dataset ({dataset_path}):\n{error}")
			exit(0)

		print(f"Successfully parse {len(features['feature_0'])} rows of features & {len(targets)} targets")
		return self.features, self.targets

	def normalize(self):

		try:
			data = {}

			if self.normalization_data:
				
				for (feature, column), (_min, _max) in zip(self.features.items(), self.normalization_data):
					data[feature] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column]

			else:
				for feature, column in self.features.items():
					_min = min(column)
					_max = max(column)
					self.normalization_data.append([_min, _max])
					data[feature] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column]

			self.features = data
			return data

		except Exception as error:
			print(f"[DataProcessing ERROR] Error while normalize data:\n{error}")
			exit(0)

	def get_data(self, binary_targets=[]):

		if not self.features:
			print(f"[DataProcessing ERROR] No features found in get_data() method.")
			return None, None

		features = np.array([np.array(features) for features in zip(*list(self.features.values()))])
		if binary_targets:
			targets = self.binary_targets_to_np(binary_targets[0], binary_targets[1])
		else:
			targets = np.array(self.targets)
		return features, targets

	def binary_targets_to_np(self, zero, one):

		targets = np.zeros((len(self.targets), 2))
		for i, label in enumerate(self.targets):
			# print(f"i={i} / label={label} / {zero} / {one}")
			if label == zero:
				targets[i] = np.array([0, 1])
			elif label == one:
				targets[i] = np.array([1, 0])
			else:
				targets[i] = np.nan
		return targets

	def save_data(self, file_path, normalization=False, standardization=False):

		try:
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

		except Exception as error:
			print(f"[DataProcessing ERROR] Cannot save normalization or standardization data: {error}")

	def load_data(self, file_path, normalization=False, standardization=False):

		try:
			with open(file_path, 'r') as f:
				data = f.read()

				if normalization:
					self.normalization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
					print(f"normalization_data: {self.normalization_data}")

				if standardization:
					self.standardization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
					print(f"standardization_data: {self.standardization_data}")

		except Exception as error:
			print(f"[DataProcessing ERROR] Cannot load normalization or standardization data: {error}")
