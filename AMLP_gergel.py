#Implementation of A-MLP with Gergel design

import numpy as np
import matplotlib.pyplot as plt
import random

class AAN(): #artificial astrocyte network
 
    # def __init__(self,size = (1,2),decay_rate = [0.8,0.7], threshold = [0.8,0.8], initial_act = [0.3,0.3]):
    def __init__(self,size = (2,10),decay_rate = 0.8, threshold = 0.8, initial_act = 0.0, stable = True, weight=0.5, learning_duration = 2, backprop_status= False):
        self.size = size
        self.input = np.zeros(size) #input from synapse
        self.output = np.zeros(size) #astrotrasmitter back into synapse
        self.activity = np.zeros(size) #holds activations
        self.act_history = np.zeros(size) #holds all activation history
        self.act_count = 0 #keeps track for average
        self.decay_rate = np.ones(size) #decays inner activation
        self.threshold = np.ones(size) #threshold for astrocyte to register activity
        self.weight_limit = 5
        self.stable_decay = stable
        self.stable_thresh = stable
        self.stable_initial = stable
        self.stable_weight = stable
        self.weights = np.ones(size) #weights
        self.learning_dur = learning_duration
        self.backprop = backprop_status

        # self.trial_count = 0 #keep track of trials for learning
        self.thresh_learn = 0.1
        decay_list = []
        thresh_list = []
        act_list = []
        weight_list = []
        if self.stable_decay == True: #added in so I can eventually use with genetic algorithm that will set mutliple decays
            for row in range(0,len(self.input)): #this will be deleted when decay_rate, treshhold, act_list are all entered as lists/ random
                # print('self input',self.input[row])
                for a in range(0,len(self.input[row])):
                    decay_list.append(decay_rate) #append random value here if want to randomize parameters

        if self.stable_thresh == True:
            for row in range(0,len(self.input)):
                for a in range(0,len(self.input[row])):
                    thresh_list.append(threshold)
        # print('threshlist',self.threshold)

        if self.stable_initial == True:
            for row in range(0,len(self.input)):
                for a in range(0,len(self.input[row])):
                    act_list.append(initial_act)
                
        if self.stable_weight == True: #added in so I can eventually use with genetic algorithm that will set mutliple decays
            for row in range(0,len(self.input)): #this will be deleted when decay_rate, treshhold, act_list are all entered as lists/ random
                # print('self input',self.input[row])
                for n in range(0,len(self.input[row])):
                    weight_list.append(weight) #append random value here if want to randomize parameters
                
        self.genes = [decay_list, thresh_list, act_list, weight_list]
        # print('genes',self.genes)

    def set_parameters(self):
        # a = 0 #iterator
        #a = 0 for decay rate
        # print('len input',len(self.input))
        # print('shape input',np.shape(self.input))
        for row in range(0,len(self.input)):
            a = 0
            for i in range(0,len(self.decay_rate[row])):
                # print('decay rate',self.decay_rate[row])
                # print('genes',self.genes)
                # print('row',row)
                # print('decay arra', self.decay_rate)
                self.decay_rate[row][i] = self.genes[a][i]

            a +=1 # threshold
            for j in range(0, len(self.threshold[row])):
                # print(a)
                # print('thresh',self.genes[a])
                self.threshold[row][j] = self.genes[a][j]

            a +=1 #initial activity
            for k in range(0,len(self.activity[row])):
                self.activity[row][k] = self.genes[a][k]
            
            a+=1
            for w in range(0,len(self.weights[row])):
                self.weights[row][w] = self.genes[a][w]
    
    def apply_limits(self):
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])): 
                if self.weights[row][astro] > self.weight_limit:
                    self.weights[row][astro] = self.weight_limit
                elif self.weights[row][astro] < (-1 * self.weight_limit):
                    self.weights[row][astro] = (-1 *self.weight_limit)


    def compute_activation(self):
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                # print('particular input',self.input[row][astro])
                # print('particular threshold',self.threshold[row][astro])
                if self.input[row][astro] >= self.threshold[row][astro]:
                    # print('input greater than threshold')
                    self.activity[row][astro] = 1
                else:
                    self.activity[row][astro] = (self.activity[row][astro] * self.decay_rate[row][astro])
        if self.backprop == True:
            self.apply_limits()

    def compute_activation_theta(self):
        self.act_count +=1 #keeps track of how many times compute activation has been called for the averaging process
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                if self.input[row][astro] >= self.threshold[row][astro]:
                    self.activity[row][astro] = 1
                else:
                    self.activity[row][astro] = self.activity[row][astro] * self.decay_rate[row][astro]

                self.act_history[row][astro] += self.activity[row][astro]
                
                # print('lenth act history',len(self.act_history[row][astro]))
                self.threshold[row][astro] = (self.act_history[row][astro] / self.act_count)
                # self.threshold[row][astro] = self.threshold[row][astro] + self.thresh_learn * (self.activity[row][astro] - self.threshold[row][astro])
        if self.backprop == True:
            self.apply_limits()

    def compute_activation_decay(self):
        self.act_count +=1
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                if self.input[row][astro] >= self.threshold[row][astro]:
                    self.activity[row][astro] = 1
                else:
                    self.activity[row][astro] = self.activity[row][astro] * self.decay_rate[row][astro]

                self.act_history[row][astro] += self.activity[row][astro]
                
                # print('lenth act history',len(self.act_history[row][astro]))
                self.decay_rate[row][astro] = 1 - (self.act_history[row][astro] / self.act_count)
                # self.threshold[row][astro] = self.activity[row][astro]
                # self.decay_rate[row][astro] = 1 - (self.activity[row][astro])
        if self.backprop == True:
            self.apply_limits()

    def compute_activation_all(self):
        self.act_count +=1
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                if self.input[row][astro] >= self.threshold[row][astro]:
                    self.activity[row][astro] = 1
                else:
                    self.activity[row][astro] = self.activity[row][astro] * self.decay_rate[row][astro]
                # self.threshold[row][astro] = self.activity[row][astro]
                self.act_history[row][astro] += self.activity[row][astro]
                
                # print('lenth act history',len(self.act_history[row][astro]))
                average_act = (self.act_history[row][astro] / self.act_count)
                self.decay_rate[row][astro] = 1 - (average_act)
                self.threshold[row][astro] = average_act
        if self.backprop == True:
            self.apply_limits()
    
    def show_parameters(self, act_histogram = False, bar=False,histograms=True):

        x_range = 0
        weightlist = []
        thres_list = []
        dec_list = []
        for row in range(0,len(self.input)):
            for astro in range(0,len(self.input[row])):
                weightlist.append(self.weights[row][astro])
                thres_list.append(self.threshold[row][astro])
                dec_list.append(self.decay_rate[row][astro])
                x_range +=1
                
        if bar == True:
            plt.subplot(1,3,1)
            plt.title('Astrocytic weights')
            plt.bar(x = range(0,x_range),height=weightlist)
    
            plt.subplot(1,3,2)
            plt.title('Astrocytic Thresholds')
            plt.bar(height = thres_list, x = range(x_range))
            # plt.legend

            plt.subplot(1,3,3)
            plt.title('Astrocytic Decay')
            plt.bar(height = dec_list, x = range(x_range))
            # plt.legend
            plt.show()

        #plot distribution of activity
        if act_histogram == True:
            activities = []
            # for act in self.activity
            for row in range(0,len(self.input)):
                for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                    activities.append(self.activity[row][astro])
                
            # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
            num_bins = 50
            plt.title('Distribution of astrocyte activity')
            n, bins, patches = plt.hist(activities, num_bins, facecolor='blue', alpha=0.5)
            plt.xlabel('Astrocyte Activation')
            plt.ylabel('Frequency')
            plt.show()

        if histograms == True:

            activities = []
            # for act in self.activity
            for row in range(0,len(self.input)):
                for astro in range(0,len(self.input[row])): #input for each n should be the activation of that neuron
                    activities.append(self.activity[row][astro])
            
                
            # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
            num_bins = 45
            
            plt.subplot(1,4,1)
            plt.suptitle('Distribution of Astro parameters')
            plt.title('Weights')
            # plt.bar(x = range(0,x_range),height=weightlist)

            # plt.title('Distribution of astrocyte activity')
            n, bins, patches = plt.hist(weightlist, num_bins, facecolor='blue', alpha=0.5)
            # plt.xlabel('Astrocyte Weight')
            plt.ylabel('Frequency')
            # plt.show()

            plt.subplot(1,4,2)
            plt.title('Threshold')
            # plt.bar(x = range(0,x_range),height=weightlist)

            # plt.title('Distribution of astrocyte activity')
            n, bins, patches = plt.hist(thres_list, num_bins, facecolor='blue', alpha=0.5)
            # plt.xlabel('Astrocytic Treshold')
            # plt.ylabel('Frequency')
            # plt.show()

            plt.subplot(1,4,3)
            plt.title('Decay Rate')
            # plt.bar(x = range(0,x_range),height=weightlist)

            # plt.title('Distribution of astrocyte decay rate')
            n, bins, patches = plt.hist(dec_list, num_bins, facecolor='blue', alpha=0.5)
            # plt.xlabel('Astrocyte Decay Rate')
            # plt.ylabel('Frequency')
            # plt.show()

            plt.subplot(1,4,4)
            plt.title('Activity')
            # plt.bar(x = range(0,x_range),height=weightlist)

            # plt.title('Distribution of astrocyte decay rate')
            n, bins, patches = plt.hist(activities, num_bins, facecolor='blue', alpha=0.5)
            # plt.xlabel('Astrocyte decay rate')
            # plt.ylabel('Frequency')
            plt.show()


if __name__ == "__main__":
    anne = AAN(initial_act=0)
    anne.set_parameters()
    print('anne threshold',anne.threshold)
    print(anne.activity)
    anne.compute_activation()
    print(anne.activity)
    anne.compute_activation()
    print(anne.activity)

