import sys
import json
import annpy
import random
import numpy as np

loss = "BinaryCrossEntropy"
monitored_loss = f'val_{loss}'
n_seed_search = 10

def parsing(dataset_path, seeds_path=None):

	data = annpy.parsing.DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	tts_seed_ = None
	try:
		with open(seeds_path, 'r') as f:
			lines = [elem for elem in f.read().split('\n') if elem and elem[0] == '{']

			global best_loss_file
			best_loss_file = 42
			for line in lines:

				# print(f"line {type(line)}: {line}")
				line = json.loads(line)
				if line.get(monitored_loss, None) < best_loss_file:
					best_loss_file = line.get(monitored_loss, None)
					tts_seed_ = line.get('tts_seed', None)

			print(f"end parsing, seed: {type(best_loss_file)}, loss: {best_loss_file}\n")

	except Exception as e:
		print(e)
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], tts_seed_

def get_model():

	model = annpy.models.SequentialModel(
		input_shape=layers_shp[0],
		name="First model",
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
		# activation="Sigmoid",
		activation="Softmax",
	))

	model.compile(
		loss=loss,
		optimizer="Adam"
	)
	return model

def get_model_train():

	model = get_model()

	early_stopping = annpy.callbacks.EarlyStopping(
		model=model,
		monitor=monitored_loss,
		patience=15,
	)

	logs = model.fit(
		features,
		targets,
		epochs=200,
		batch_size=32,
		callbacks=[early_stopping],
		verbose=False,
		print_graph=False
	)

	print(f"Fit result: {logs[monitored_loss][-1]}")
	logs = early_stopping.get_best_metrics()
	print(f"Fit best  : {logs}\n")
	return model, logs


# Protection & Parsing
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

features, targets, input_shape, tts_seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
layers_shp = (input_shape, 64, 32, 2)

if not tts_seed:
	tts_seed = np.random.get_state() ####

# Model search
best_model = None
best_loss = 42

i = 0
while i < n_seed_search:

	randn = random.choice(tts_seed[1]) % 1000000
	np.random.seed(randn)

	print(f"-- Seed {i+1}/{n_seed_search} -- seed={randn}")
	model, logs = get_model_train()

	seed_loss = logs[monitored_loss]

	if seed_loss < best_loss:
		best_loss = seed_loss
		best_model = model

	# print(f"- Average loss: {seed_loss}\t-- Best loss: {best_loss}\n")
	i += 1

print(f"Average loss of the best seed ({n_seed_search} tries): {best_loss}")

if best_loss < best_loss_file:

	seed = best_model.get_seeds()['w_seed']
	with open("ressources/seeds.txt", 'a') as f:
		seed = list(seed)
		tts_seed = list(tts_seed)

		seed[1] = [int(n) for n in seed[1]]
		tts_seed[1] = [int(n) for n in tts_seed[1]]

		seed_dict = {
			monitored_loss: best_loss,
			'layers_shp': layers_shp,
			'seed': seed,
			'tts_seed': tts_seed,
		}
		print(f"Write best loss in file")
		json.dump(seed_dict, f)
		print('\n', file=f)

else:
	print(f"Best loss found is worst than best loss file fetch in seeds file: {best_loss} (script) > (file) {best_loss_file}")