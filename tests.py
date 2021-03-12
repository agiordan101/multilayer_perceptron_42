import sys
import plotly.express as px
from plotly.offline import plot
from DataProcessing import DataProcessing

normalization_path = "ressources/normalization.txt"

# Protection
if len(sys.argv) != 2:
	print("1 arguments needed: dataset")
	exit(1)


#Open dataset and get lines
dataset_file = open(sys.argv[1], "r")
dataset_str = dataset_file.read()

# Get all numbers
dataset_lst = [[float(x) if not x.isalpha() else x for x in line.split(',')[1:]] for line in dataset_str.split('\n')[1:-1]]

features_name = ["target", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29"]

dataset_dict = {}
for feature in features_name:
	dataset_dict[feature] = []

for features in dataset_lst:
	for feature, x in zip(features_name, features):
		dataset_dict[feature].append(x)

dataProcessing = DataProcessing(dataset_dict, columns=features_name)
dataProcessing.normalize()
dataProcessing.save_data(normalization_path, normalization=True)
dataset_np = dataProcessing.get_data()

print(dataset_np)
# print(f"{len(dataset)} features")

fig = px.scatter_matrix(dataset_dict,
						dimensions=features_name,
						color=features_name[0],
						title="Scatter matrix of all features")

# dataset_dict["target"] = [1 if target == 'M' else 0 for target in dataset_dict["target"]]
# fig = px.parallel_coordinates(dataset_dict, color="target", labels=features_name,
#                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
plot(fig)
