import numpy as np
from collections import OrderedDict


class GraphUCB:
	def __init__(self, context_dimension, arm_num, W, all_nodes, ALPHA, BETA, LAMBDA, RHO):
		self.context_dimension = context_dimension
		self.W = W
		self.arm_num = arm_num
		self.ALPHA = ALPHA
		self.BETA = BETA
		self.LAMBDA = LAMBDA
		self.RHO = RHO
		self.selected_nodes = []
		self.all_nodes = all_nodes

		A = LAMBDA * np.identity(n=context_dimension)
		self.A1s = [A] * self.arm_num
		self.b1s = [np.zeros(context_dimension)] * self.arm_num
		self.A1Invs = [np.linalg.inv(A)] * self.arm_num

		self.A2s = [A] * self.arm_num
		self.b2s = [np.zeros(context_dimension)] * self.arm_num
		self.A2Invs = [np.linalg.inv(A)] * self.arm_num

		self.thetas1 = [np.zeros(shape=context_dimension)] * self.arm_num
		self.thetas2 = [np.zeros(shape=context_dimension)] * self.arm_num

		print ("Finish Initialization")

	def getProb(self, node):
		"""
		:param alpha:
		:param node:
		:return:
		"""
		arm_index = node.cluster
		mean1 = np.dot(self.thetas1[arm_index], node.contextFeatureVector)
		var1 = np.sqrt(np.dot(np.dot(node.contextFeatureVector, self.A1Invs[arm_index]), node.contextFeatureVector))

		neighborsFeatureVectorList = self.getNeighborsFeatureVectorList(node)
		neighborsFeatureVector = self.subtract(node, neighborsFeatureVectorList)
		# neighborsFeatureVector = self.average(neighborsFeatureVectorList)

		mean2 = np.dot(self.thetas2[arm_index], neighborsFeatureVector)
		var2 = np.sqrt(np.dot(np.dot(neighborsFeatureVector, self.A2Invs[arm_index]), neighborsFeatureVector))

		# pta = (mean1 + ALPHA * var1) * self.RHO + (ALPHA * var2 + mean2) * (1 - self.RHO)  # + anomalyNeighborCount
		pta = (mean1 + self.ALPHA * var1) + (self.BETA * var2 + mean2) * self.RHO
		return pta

	def updateParameters(self, picked_node, reward):

		# indexList = self.W.T[picked_node.id]
		picked_arm_index = picked_node.cluster
		neighborsFeatureVectorList = self.getNeighborsFeatureVectorList(picked_node)
		neighborsFeatureVector = self.average(neighborsFeatureVectorList)
		# neighborsFeatureVector = self.subtract(picked_node, neighborsFeatureVectorList)

		self.A1s[picked_arm_index] = np.add(self.A1s[picked_arm_index], np.outer(picked_node.contextFeatureVector, picked_node.contextFeatureVector), casting="unsafe")
		tmp = reward - np.dot(self.thetas2[picked_arm_index], neighborsFeatureVector) * self.RHO
		# tmp = (reward - np.dot(self.thetas2[picked_arm_index], neighborsFeatureVector) * (1 - self.RHO)) / self.RHO

		self.b1s[picked_arm_index] = np.add(self.b1s[picked_arm_index], tmp * picked_node.contextFeatureVector, casting="unsafe")
		self.A1Invs[picked_arm_index] = np.linalg.inv(self.A1s[picked_arm_index])
		self.thetas1[picked_arm_index] = np.dot(self.A1Invs[picked_arm_index], self.b1s[picked_arm_index])

		self.A2s[picked_arm_index] = np.add(self.A2s[picked_arm_index], np.outer(neighborsFeatureVector, neighborsFeatureVector), casting="unsafe")
		tmp = (reward - np.dot(self.thetas1[picked_arm_index], picked_node.contextFeatureVector)) / self.RHO
		# tmp = (reward - self.RHO * np.dot(self.thetas1[picked_arm_index], picked_node.contextFeatureVector)) / (1 - self.RHO)

		self.b2s[picked_arm_index] = np.add(self.b2s[picked_arm_index], tmp * neighborsFeatureVector, casting="unsafe")
		self.A2Invs[picked_arm_index] = np.linalg.inv(self.A2s[picked_arm_index])
		self.thetas2[picked_arm_index] = np.dot(self.A2Invs[picked_arm_index], self.b2s[picked_arm_index])

	def decide(self, nodes):

		maxPTA = float("-inf")
		picked_node = None
		# picked_arm_index = None
		# self.getCothetas(nodes)

		for id, node in enumerate(nodes):
			if id in self.selected_nodes:
				continue
			arm_pta = self.getProb(node)
			if maxPTA < arm_pta:
				picked_node = node
				maxPTA = arm_pta
		# # return a node
		self.selected_nodes.append(picked_node.id)

		#get dependent arm
		return picked_node

	def getNeighborsFeatureVectorList(self, node):
		neighborsFeatureVectorList = []
		indexList = np.nonzero(self.W[node.id])[0]
		if len(indexList) == 1:
			neighborsFeatureVectorList.append(node.contextFeatureVector)
			return neighborsFeatureVectorList

		for i in indexList:
			if self.all_nodes[i].id == node.id:
				continue
			neighborsFeatureVectorList.append(self.all_nodes[i].contextFeatureVector)

		# neighborsFeatureVector = np.mean(neighborsFeatureVectorList, axis=0)
		return neighborsFeatureVectorList

	def getNeighborsFeatureVectorMap(self, node):
		# import pdb
		# pdb.set_trace()
		neighborsFeatureVectorMap = OrderedDict({})

		neighborsFeatureVectorMap[str(node.id)] = str(node.contextFeatureVector)

		indexList = np.nonzero(self.W[node.id])[0]

		if len(indexList) == 1:
			neighborsFeatureVectorMap[str(node.id)] = str(node.contextFeatureVector)
			return neighborsFeatureVectorMap

		
		for i in indexList:
			# pdb.set_trace()
			if self.all_nodes[i].id == node.id:
				continue
			neighborsFeatureVectorMap[str(self.all_nodes[i].id)] = str(self.all_nodes[i].contextFeatureVector)

		return neighborsFeatureVectorMap

	@staticmethod
	def average(neighborsFeatureVectorList):
		neighborsFeatureVector = np.mean(neighborsFeatureVectorList, axis=0)
		return neighborsFeatureVector

	@staticmethod
	def max(neighborsFeatureVectorList):
		neighborsFeatureVector = np.amax(neighborsFeatureVectorList, axis=0)
		return neighborsFeatureVector

	@staticmethod
	def sum(neighborsFeatureVectorList):
		neighborsFeatureVector = np.sum(neighborsFeatureVectorList, axis=0)
		return neighborsFeatureVector

	@staticmethod
	def subtract(node, neighborsFeatureVectorList):
		neighborsFeatureVector = np.mean(neighborsFeatureVectorList, axis=0)
		neighborsFeatureVector = np.abs(node.contextFeatureVector - neighborsFeatureVector)
		return neighborsFeatureVector