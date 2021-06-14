import numpy as np
from annpy.losses.Loss import Loss

class MSE(Loss):

	def __init__(self):
		super().__init__()
		self.name = "MSE"

	def compute(self, predictions, targets):
		ret = np.mean(np.mean(np.square(targets - predictions), axis=0))
		# print(f"prediction {prediction.shape}: {prediction}")
		# print(f"target {target.shape}: {target}")
		# print(f"MSE {ret.shape}: {ret}")
		# exit(0)
		return ret

	# def get_mem_len_append(self, predictions, targets):
	# 	return 1
	
	def derivate(self, prediction, target):
		return prediction - target

	def summary(self):
		print(f"Loss:\t\tannpy.losses.MSE")
