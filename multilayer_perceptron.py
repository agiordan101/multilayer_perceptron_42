import sys
import json
# import annpy
import annpy

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

	return features[0].shape[0], features, targets, seed, tts_seed
	# return features, targets, features[0].shape[0], seed, tts_seed

def get_model(layers_shp, seed=None, tts_seed=None):

	model = annpy.models.SequentialModel(
		input_shape=layers_shp[0],
		name="MLP_42",
		seed=seed,
		tts_seed=tts_seed
	)
	model.add(annpy.layers.FullyConnected(
		layers_shp[1],
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		layers_shp[2],
		activation="tanh",
	))
	model.add(annpy.layers.FullyConnected(
		layers_shp[3],
		activation="Softmax",
	))
	model.compile(
		loss=loss,
		optimizer="Adam",
	)
	return model

def get_model_train(layers_shp, features, targets, seed=None, tts_seed=None):

	model = get_model(layers_shp, seed, tts_seed)

	early_stopping = annpy.callbacks.EarlyStopping(
		model=model,
		monitor=monitored_loss,
		patience=15,
	)

	logs = model.fit(
		features,
		targets,
		epochs=500,
		batch_size=32,
		callbacks=[early_stopping],
		verbose=False,
		print_graph=False
	)
	
	logs = {key: mem[-1] for key, mem in logs.items()}
	print(f"End fit result: {logs}")

	best_logs = early_stopping.get_best_metrics()
	print(f"Best result   : {logs}")

	return model, logs, best_logs



# Protection
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

# Parsing
input_shape, data = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
# features, targets, input_shape, seed, tts_seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

layers_shp = (input_shape, 64, 32, 2)
# layers_shp = (input_shape, 64, 32, 2)

# MLP
model, logs, best_logs = get_model_train(layers_shp, *data)
# model, logs = get_model_train(layers_shp, seed, tts_seed)
