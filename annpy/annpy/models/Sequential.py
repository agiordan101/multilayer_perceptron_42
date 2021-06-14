# import annpy
# from annpy.models.Model import Model
# from annpy.layers.Layer import Layer

# import numpy as np

# class sequential(Model):

# 	def __init__(self,
# 					input_shape=None,
# 					input_layer=None,
# 					name="Default models name"):

# 		super().__init__(input_shape, input_layer, name)

# 		if input_layer:
# 			self.sequence = [input_layer]
# 		else:
# 			self.sequence = []

# 	def __str__(self):
# 		return "Sequential"

# 	def add(self, layer):

# 		if issubclass(type(layer), Layer):
# 			# Add Object into sequential model
# 			layer.set_layer_index(len(self.sequence))
# 			self.sequence.append(layer)
# 		else:
# 			raise Exception(f"[annpy error]:Object {layer} is not a child of abstact class {Layer}")

# 	def compile(self,
# 				loss="MSE",
# 				optimizer="Adam",
# 				metrics=[]):
		
# 		# Save loss, optimizer and metrics
# 		super().compile(loss, optimizer, metrics)

# 		# input_shape handler
# 		if self.input_shape:
# 			pass
# 		elif self.sequence[0].input_shape:
# 			self.input_shape = self.sequence[0].input_shape
# 		else:
# 			raise Exception(f"[annpy error]:[ERROR] {self} input_shape of layer 0 missing")
# 		input_shape = self.input_shape

# 		# self.weights:	[[w0, b0], [..., ...], [wn, bn]]
# 		self.weights = []
# 		for layer in self.sequence:

# 			weights = layer.compile(input_shape)
# 			self.weights.append(weights)

# 			# Save next input shape
# 			input_shape = layer.output_shape

# 			# Add matrix for optimizer math
# 			self.optimizer.add(weights)

# 		# Optimizers maths
# 		self.optimizer.compile()

# 		# Make reverse list for fitting method
# 		self.sequence_rev = self.sequence.copy()
# 		self.sequence_rev.reverse()

# 	def forward(self, inputs):

# 		# print(f"Input shape={inputs.shape}")
# 		# inputs = np.insert(inputs, len(inputs), 1, axis=len(inputs.shape) - 1)
# 		# print(inputs)
# 		for layer in self.sequence:
# 			inputs = layer.forward(inputs)

# 		return inputs

# 	def fit(self,
# 			train_features,
# 			train_targets,
# 			batch_size=42,
# 			epochs=420,
# 			callbacks=[],
# 			val_features=None,
# 			val_targets=None,
# 			val_percent=0.2,
# 			verbose=True):

# 		# Callbacks TRAIN begin
# 		for cb in callbacks:
# 			cb.on_train_begin(model=self)

# 		super().fit(train_features, train_targets, batch_size, epochs, callbacks, val_features, val_targets, val_percent, verbose)

# 		for epoch in range(epochs):

# 			print(f"EPOCH {epoch}")

# 			# Callbacks EPOCH begin
# 			for cb in callbacks:
# 				cb.on_epoch_begin()

# 			# Dataset shuffle + split
# 			batchs = self.batchs_split()
# 			# batchs = self.batchs_split(train_features, train_targets, self.batch_split)
			
# 			# for b in batchs:
# 			# 	print(f"batch shape : {b[0].shape}")
# 			# print(f"{len(batchs) - 1} batches with shape : {batchs[0][0].shape}")
# 			# print(f"1 batch with shape : {batchs[-1][0].shape}")
			
# 			# print(list(batchs))
# 			# for step, (features, target) in enumerate(batchs):
# 			for step, data in enumerate(batchs):

# 				if verbose:
# 					print(f"STEP={step}/{self.n_batch - 1}")
# 					# print(f"STEP={step}/{self.n_batch - 1}\tloss: {self.loss.get_result()}")

# 				# Callbacks BATCH begin
# 				for cb in callbacks:
# 					cb.on_batch_begin()

# 				features, target = data

# 				# Prediction
# 				prediction = self.forward(features)

# 				# Metrics actualisation
# 				for metric in self.metrics.values():
# 					# print(f"train={self.val_metrics_on} -> {metric.name}")
# 					metric(prediction, target)

# 				# Backpropagation
# 				dx = self.loss.derivate(prediction, target)
# 				gradients = [0, 0]
# 				self.optimizer.gradients = []
# 				for layer in self.sequence_rev:
# 					# print(f"LAYER {layer.layer_index}")
# 					dx, gradients = layer.backward(dx)
# 					# gradients = layer.backward(gradients[0])
# 					self.optimizer.gradients.append(gradients)

# 				# Optimizer
# 				self.optimizer.apply_gradients(self.weights)
# 				# exit(0)

# 				# Callbacks BATCH end
# 				for cb in callbacks:
# 					cb.on_batch_end()

# 			# self.debug.append(list(self.metrics.values())[0].get_result())

# 			self.evaluate(self, self.val_features, self.val_targets)

# 			# Get total metrics data of this epoch
# 			print(f"Metrics: {self.get_metrics_logs()}")
# 			print(f"\n-------------------------\n")

# 			# Callbacks EPOCH end
# 			for cb in callbacks:
# 				cb.on_epoch_end(
# 					model=self,
# 					metrics=self.metrics
# 				)

# 			# Save in mem & Reset metrics values
# 			self.reset_metrics(save=True)

# 			if self.stop_trainning:
# 				break

# 		self.print_graph()

# 		# Callbacks TRAIN end
# 		for cb in callbacks:
# 			cb.on_train_end()

# 		return self.loss.get_mem()[-1], self.accuracy.get_mem()[-1]


# 	def summary(self, only_model_summary=True):
		
# 		super().summary(only_model_summary=only_model_summary)

# 		print(f"Input shape:\t{self.input_shape}")
# 		print(f"Output shape:\t{self.sequence[-1].output_shape}\n")
# 		for layer in self.sequence:
# 			layer.summary()

# 		if only_model_summary:
# 			print(f"-------------------\n")

# 	def deepsummary(self, model_summary=True):
		
# 		if model_summary:
# 			self.summary(only_model_summary=False)

# 		for i, layer in enumerate(self.weights):
# 			print(f"\nLayer {i}:\n")
# 			print(f"Weights {layer[0].shape}:\n{layer[0]}\n")
# 			print(f"Bias {layer[1].shape}:\n{layer[1]}\n")

# 		print(f"-------------------\n")
