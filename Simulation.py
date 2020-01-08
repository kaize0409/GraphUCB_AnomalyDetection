import datetime
import os
import random
import scipy.io
from GraphUCB import GraphUCB


class Node:
    def __init__(self, id, label, cluster, contextFeatureVector=None):
        self.id = id
        self.contextFeatureVector = contextFeatureVector
        self.label = label
        self.cluster = cluster


class Simulation(object):
    def __init__(self, iterations, algorithms, training_iters, nodes, arm_num, graph):
        """

        :param iterations:
        :param algorithms:
        """
        self.iterations = iterations
        self.algorithms = algorithms
        self.start_time = datetime.datetime.now()
        self.training_iters = training_iters
        self.arm_num = arm_num
        self.graph = graph
        self.nodes = nodes
        self.selected_nodes = []
        # self.all_nodes = all_nodes
        # self.two_hop = np.dot(self.graph, self.graph)

    def run(self):
        """
        :return:
        """
        timeRun = self.start_time.strftime('_%m_%d_%H_%M')
        regretFilePath = os.path.join("./Out", "AccRegret" + timeRun + ".csv")
        with open(regretFilePath, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write('\n')
            f.write('0,' + ','.join(['0' for alg_name in algorithms.keys()]))
            f.write('\n')
        f.close()

        optimalReward = 1
        algRegret = {}
        batchRegret = {}


        #training
        counter = {}
        for iter in range(1, self.training_iters + 1):
            print (iter)
            for alg_name, alg in self.algorithms.items():
                if alg_name not in counter:
                    counter[alg_name] = 0

                pickedNode = alg.decide(self.nodes)
                # self.selected_nodes.append(pickedNode)
                reward = self.getReward(pickedNode)  # + noise
                alg.updateParameters(pickedNode, reward)

                if reward == 1:
                    counter[alg_name] += 1

        for alg, cnt in counter.items():
            print (alg + ": " + str(cnt))

        for alg_name, alg in self.algorithms.items():
            alg.selected_nodes = []
            algRegret[alg_name] = []
            batchRegret[alg_name] = []

        # testing iteration
        for iter in range(1, self.iterations + 1):
            print (iter)
            for alg_name, alg in self.algorithms.items():

                pickedNode = alg.decide(self.nodes)
                reward = self.getReward(pickedNode)  # + noise
                alg.updateParameters(pickedNode, reward)

                regret = optimalReward - reward
                algRegret[alg_name].append(regret)
            if iter % 5 == 0:
                for alg_name in self.algorithms.keys():
                    batchRegret[alg_name].append(sum(algRegret[alg_name]))
                with open(regretFilePath, 'a+') as f:
                    f.write(str(iter))
                    f.write(',' + ','.join([str(batchRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')
                f.close()

    @staticmethod
    def getReward(arm):
        return 1 if arm.label is True else 0


def getTrainAndAll(labels, attributes, training_true_num):

    nodes = []
    index_list1 = []
    index_list2 = []
    for id, label in enumerate(labels):
        attribute = attributes[id].T
        if label[0] == 1:
            node = Node(id, True, attribute.reshape(len(attribute)))
            index_list1.append(id)
        else:
            node = Node(id, False, attribute.reshape(len(attribute)))
            index_list2.append(id)
        nodes.append(node)

    training_index_list = random.sample(index_list1, training_true_num)
    training_index_list.extend(random.sample(index_list2, training_true_num * 10))
    training_nodes = [nodes[i] for i in training_index_list]
    return training_nodes, nodes


def getClasses(labels, attributes, classes):
    class_list = []
    for cla in classes:
        if cla[0] not in class_list:
            class_list.append(cla[0])

    class_num = len(class_list)

    class_node_list = []
    for i in class_list:
        index_list = []
        for node_id, class_id in enumerate(classes):
            if class_id == i:
                index_list.append(node_id)

        node_list = []
        for index in index_list:
            if labels[index] == 1:
                node = Node(index, True, attributes[index])
            else:
                node = Node(index, False, attributes[index])
            node_list.append(node)
        class_node_list.append(node_list)

    return class_node_list, class_num


def getNodes(labels, attributes, classes):
    node_list = []
    for id, label in enumerate(labels):
        if labels[id] == 1:
            node = Node(id, True, classes[id][0] - 1, attributes[id])
        else:
            node = Node(id, False, classes[id][0] - 1, attributes[id])
        node_list.append(node)
    return node_list


if __name__ == '__main__':

    iterations = 250
    training_iters = 0
    algorithms = {}

    #  get data
    data = scipy.io.loadmat("BlogCatalog.mat")
    labels = data["Label"]
    attributes = data["Attributes"]
    graph = data["Network"].toarray()
    classes = data["Class"]

    class_set = set([i[0] for i in classes])
    arm_num = len(class_set)
    print("arm_num = " + str(arm_num))

    context_dimension = len(attributes[0])
    print("context_dimension = " + str(context_dimension))

    all_nodes = getNodes(labels, attributes, classes)

    algorithms['GraphUCB'] = GraphUCB(context_dimension, arm_num, graph, all_nodes, ALPHA=10, LAMBDA=0.1, BETA=0.1, RHO=10)

    simExperiment = Simulation(iterations, algorithms, training_iters, all_nodes, arm_num, graph)
    simExperiment.run()

