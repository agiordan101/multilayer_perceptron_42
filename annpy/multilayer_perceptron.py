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
metrics = ["RangeAccuracy"]

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
				if line.get(monitored_loss) < best_loss_file:
					best_loss_file = line.get(monitored_loss)
					seed = line.get('seed')
					tts_seed = line.get('tts_seed')

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
		metrics=metrics
	)
	return model

def get_model_train(model_shp, features, targets, seed=None, tts_seed=None, optimizer='Adam', optimizer_kwargs={}):

	# model = get_model(model_shp, None, tts_seed, optimizer, optimizer_kwargs)
	model = get_model(model_shp, seed, tts_seed, optimizer, optimizer_kwargs)

	# model.deepsummary()
	# return
	early_stopping = None
	early_stopping = annpy.callbacks.EarlyStopping(
		model=model,
		# monitor="val_RangeAccuracy",
		monitor=monitored_loss,
		patience=200,
	)

	logs = model.fit(
		features,
		targets,
		epochs=500,
		batch_size=128,
		callbacks=[early_stopping] if early_stopping else [],
		val_percent=0.2,
		verbose=False,
		print_graph=False
	)

	end_fit_logs = {key: mem[-1] for key, mem in logs.items()}
	print(f"\nEnd fit {optimizer} result: {end_fit_logs}")

	if early_stopping:
		best_logs = early_stopping.get_best_metrics()
		print(f"End fit {optimizer}, best : {best_logs}")
	else:
		best_logs = None

	# model.deepsummary()
	# print(f"models weights: {model}")
	return model, logs, best_logs

def	benchmark(model_shp, data):

	models = [
		('SGD', get_model_train(model_shp, *data, optimizer='SGD')),
		('RMSProp', get_model_train(model_shp, *data, optimizer='RMSProp')),
		('RMSProp_momentum', get_model_train(model_shp, *data, optimizer='RMSProp', optimizer_kwargs={'momentum': 0.986})),
		('Adam', get_model_train(model_shp, *data, optimizer='Adam')),
	]

	max_epoch = max(len(logs[1][monitored_loss]) for metric, logs in models)
	x_axis = np.array(list(range(max_epoch)))
	subject_goal_line = [0.08] * max_epoch

	plt.plot(x_axis, subject_goal_line)
	for i, model in enumerate(models):
		metric, mem = model
		# print(type(mem[1]), type(mem[1]))
		# plt.plot(x_axis[:len(mem[1][loss])], mem[1][loss], label=metric)
		plt.plot(x_axis[:len(mem[1][monitored_loss])], mem[1][monitored_loss], label=metric)

	plt.title('Metrics based on Optimizers')
	plt.xlabel('Epochs')
	plt.ylabel('Losses')
	plt.legend()
	plt.show()

parser = argparse.ArgumentParser(description='Multilayer-Perceptron')
parser.add_argument('-dataset', required=True, dest='dataset_path')
parser.add_argument('-seeds', dest='seeds_path', default=None)

parser.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', default=False)

args = vars(parser.parse_args())


# Parsing
input_shape, data = parsing(args['dataset_path'], args['seeds_path'])

model_shp = (input_shape, 64, 32, 2)

if args['benchmark']:
	print(f"Optimizers benchmark")

	benchmark(model_shp, data)


if args['train']:
	print(f"Train new MLP & save weights")

	model, logs, best_logs = get_model_train(model_shp, *data)
	model.save_weights(folder_path="./ressources", file_name="mlp_adam_train")


if args['test']:
	print(f"Load and test model")

	model = annpy.models.SequentialModel.load_model(args['test'])
	model.compile_metrics(loss, metrics)
	logs = model.evaluate(data[0], data[1], return_stats=True)
	print(f"Model loaded: {logs}")
