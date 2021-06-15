from abc import ABCMeta, abstractmethod
from annpy.metrics.Metric import Metric

# class Loss(metaclass=ABCMeta):
class Loss(Metric):

	@abstractmethod
	def derivate(self, prediction, target):
		pass

	def get_variation_goal(self):
		return 'min'
