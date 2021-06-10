import sys
import json
import annpy
import argparse
import matplotlib

matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
# import plotly.express as px
# from annpy import annpy
import pandas as pd
import matplotlib as mpl
import numpy as np

loss = "BinaryCrossEntropy"
monitored_loss = f'val_{loss}'

def parsing(dataset_path, seeds_path=None):

	data = annpy.parsing.DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	seed = None
	tts_seed = None
	try:
		with open(seeds_path, 'r') as f:
			lines = [elem for elem in f.read().split('\n') if elem and elem[0] == '{']

			best_loss_file = 42
			for line in lines:

				# print(f"line {type(line)}: {line}")
				line = json.loads(line)
				if line.get(monitored_loss, None) < best_loss_file:
					best_loss_file = line.get(monitored_loss, None)
					seed = line.get('seed', None)
					tts_seed = line.get('tts_seed', None)

			print(f"End parsing, seed: {bool(seed)}, tts_seed: {bool(tts_seed)}\n")

	except:
		print(f"No seed found.\n")

	return features[0].shape[0], (features, targets, seed, tts_seed)
	# return features, targets, features[0].shape[0], seed, tts_seed

def get_model(model_shp, seed=None, tts_seed=None, optimizer='Adam', optimizer_kwargs={}):

	model = annpy.models.SequentialModel(
		input_shape=model_shp[0],
		name="MLP_42",
		seed=seed,
		tts_seed=tts_seed
	)
	model.add(annpy.layers.FullyConnected(
		model_shp[1],
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		model_shp[2],
		activation="tanh",
	))
	model.add(annpy.layers.FullyConnected(
		model_shp[3],
		activation="Softmax",
	))
	model.compile(
		loss=loss,
		optimizer=optimizer,
		optimizer_kwargs=optimizer_kwargs,
		metrics=["RangeAccuracy"]
	)
	return model

def get_model_train(model_shp, features, targets, seed=None, tts_seed=None, optimizer='Adam', optimizer_kwargs={}):

	model = get_model(model_shp, seed, tts_seed, optimizer, optimizer_kwargs)

	early_stopping = None
	early_stopping = annpy.callbacks.EarlyStopping(
		model=model,
		# monitor="val_RangeAccuracy",
		monitor=monitored_loss,
		patience=15,
	)

	logs = model.fit(
		features,
		targets,
		epochs=100,
		batch_size=32,
		callbacks=[early_stopping] if early_stopping else [],
		val_percent=0.2,
		verbose=False,
		print_graph=False
	)

	end_fit_logs = {key: mem[-1] for key, mem in logs.items()}
	print(f"End fit result: {end_fit_logs}")

	if early_stopping:
		best_logs = early_stopping.get_best_metrics()
		print(f"End fit, best : {best_logs}")
	else:
		best_logs = None


	return model, logs, best_logs

def	benchmark(model_shp, data):

	models = [
		('Adam', get_model_train(model_shp, *data, optimizer='Adam')[1][monitored_loss]),
		('SGD', get_model_train(model_shp, *data, optimizer='SGD')[1][monitored_loss]),
		('SGD_momentum', get_model_train(model_shp, *data, optimizer='SGD', optimizer_kwargs={'momentum': 0.98})[1][monitored_loss]),
		('RMSProp', get_model_train(model_shp, *data, optimizer='RMSProp')[1][monitored_loss]),
	]

	max_epoch = max(len(mem) for metric, mem in models)
	x_axis = np.array(list(range(max_epoch)))
	subject_goal_line = [0.08] * max_epoch

	fig, axs = plt.subplots(2, 2)
	fig.suptitle('Metrics based on Optimizers')

	for i, model in enumerate(models):
		metric, mem = model
		axs[i//2, i%2].set_title(metric)
		axs[i//2, i%2].plot(x_axis, subject_goal_line)
		axs[i//2, i%2].plot(x_axis, mem + [mem[-1]] * (max_epoch - len(mem)))
		axs[i//2, i%2].set_xlim([0, max_epoch])
		axs[i//2, i%2].set_ylim([0, 1])

	# fig.legend()
	plt.xlim(0, max_epoch)
	plt.ylim(0, 1)
	plt.show()
	print(f"END")

parser = argparse.ArgumentParser(description='Multilayer-Perceptron')
parser.add_argument('-dataset', required=True, dest='dataset_path')
parser.add_argument('-seeds', dest='seeds_path', default=None)

parser.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)

args = vars(parser.parse_args())

print(args['dataset_path'], args['seeds_path'], args['benchmark'])


# Parsing
input_shape, data = parsing(args['dataset_path'], args['seeds_path'])
# features, targets, input_shape, seed, tts_seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

model_shp = (input_shape, 64, 32, 2)
# model_shp = (input_shape, 64, 32, 2)

if args['benchmark']:
	benchmark(model_shp, data)

else:
	# MLP
	model, logs, best_logs = get_model_train(model_shp, *data)
	# model, logs = get_model_train(model_shp, seed, tts_seed)

