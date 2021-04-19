# this is a file for creating the N-parity dataset to train a mlp

import numpy as np
import math
import matplotlib.pyplot as plt
import random
# import conx as cx

class Nparity_dataset():

    def __init__(self, N=1, amount=9):
        self.X = []
        self.Outputs = [] # 1 = even, 0 = odd
        self.train_x = []
        self.validate_x = []
        self.train_y = []
        self.validate_y = []
        self.N = N #amount of inputs per instance
        self.size = 2 ** N #this was just how they did it in the paper but it is how many integers in whole dataset

        
    def populate(self):
        for i in range(0,self.size):
            # self.X.append(np.zeros(self.size))
            instance = np.random.randint(2, size=self.N)
            self.X.append(list(instance))
            count = 0
            for j in instance:
                if j == 1:
                    count +=1
            if count % 2 == 0:
                self.Outputs.append([1])
            else:
                self.Outputs.append([0])
            
        print('size',self.size)
    #     print(train_count)
        print('len of x',len(self.X))   
    



class four_to_one_MUX(): # [b a D C B A] rule from https://www.electronics-tutorials.ws/combination/comb_2.html

    def __init__(self, size = 1):
        self.x = []
        self.y = []
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.size = size
    
    def calc_output(self,instance):
        output = 0
        # print('instnace[0',instance[0])
        if instance[0] == 0:
            if instance[1] == 0:
                output = instance[-1]
            
            elif instance[1] == 1:
                output = instance[-2]
        
        elif instance[0] == 1:
            if instance[1] == 0:
                output = instance[3]
            elif instance[1] == 1:
                output = instance[2]
        return [output]


    def populate(self):
        for i in range(0,self.size):
            self.x.append(list(np.random.randint(2, size=6))) #fill x with random 1 and 0s
        
        for j in range(0,self.size): #calculate appropraite y from 
            instance = self.x[j]
            # print('instnace',instance)
            output = self.calc_output(instance)
            # print('output',output)
            self.y.append(output)


class two_spirals():
    def __init__(self,size = 30): #size is for amount of points for each
        self.a = []
        self.b = []
        self.x = []
        self.y = [] #holds scaler 0=A, 1=B
        self.range = size
        self.Outputs = [] #holds strings
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.text_y = []
        # self.spiral_num = spiral_num
        pass

    def spiral_xy(self, i, spiral_num): #code from https://conx.readthedocs.io/en/latest/Two-Spirals.html
        """
        Create the data for a spiral.

        Arguments:
            i runs from 0 to 96
            spiral_num is 1 or -1
        """
        φ = i/16 * math.pi
        r = 6.5 * ((104 - i)/104)
        x = (r * math.cos(φ) * spiral_num)/13 + 0.5
        y = (r * math.sin(φ) * spiral_num)/13 + 0.5
        return (x, y)

    def spiral(self,spiral_num):
        return [self.spiral_xy(i, spiral_num) for i in range(self.range)] #range is the amount of points 

    def set_spirals(self):
        self.a = ["A", self.spiral(1)]
        self.b = ["B", self.spiral(-1)]
        # print('len a', len(self.a[1]))
        # print('len b', len(self.b[1]))
        #want an x with all the datapoints, and a y that matches it, but what if we do it like n parity, where the output is generated from the x, so put the x's together, mix the tuples, and then assign output
        total_list = self.a[1] + self.b[1]
        self.x = random.sample(total_list, len(total_list))
        # print('x',len(self.x))
        for dp in range(0,len(self.x)):
            if self.x[dp] in self.a[1]:
                self.Outputs.append(self.a[0])
            elif self.x[dp] in self.b[1]:
                self.Outputs.append(self.b[0])
            else:
                print('this data point is in neither list')

    def string_toscaler(self):
        datapoint = []
        for value in self.Outputs:
            if value == 'A':
                datapoint.append([0])
            elif value == 'B':
                datapoint.append([1])
            else:
                'unknown value'
        self.y = datapoint
        return datapoint
        # print('len y',self.y[0:5])

        # self.y = en y 

    def plot_spirals(self):
        # print(a[1])
        x_list = []
        y_list = []
        bx_list = []
        by_list = []
        for x in range(0,len(self.a[1])):
            x_list.append(self.a[1][x][0])
            y_list.append(self.a[1][x][1])
            bx_list.append(self.b[1][x][0])
            by_list.append(self.b[1][x][1])
        plt.scatter(x_list,y_list, label= self.a[0])
        plt.scatter(bx_list,by_list, label=self.b[0])
        plt.legend()
        plt.show()

    def test_train_split(self,train_proportion = 0.7): #0.7 = 70% train, 30% test
        train_length = int(len(self.x) * train_proportion)
        self.train_x = self.x[:train_length]
        self.train_y = self.y[:train_length]
        self.test_x = self.x[train_length:]
        self.test_y = self.x[train_length:]
            




class Chaos_time():
    def __init__(self):
        self.phi_map = []
        pass

    def skew_map(self,a, duration, initial_val=0.43):
        # initial_value = initial_val
        φ = [initial_val] #holds y axis of time series
        for i in range(0,duration):
            # print('i',i)
            past_φ = φ[i]
            len_phi = len(φ)
            # print('past',past_φ)
            if past_φ > 1:
                past_φ = 0.99 #stuck at -1
            if past_φ == -1:
                past_φ = -0.99

            if -1 <= past_φ:
                # print('over -1')
                if past_φ <= a:
                    # print('over -1, under= a')
                    φ.append(((2*past_φ)+1-a) / 1+a)
            if a < past_φ:
                # print('overa')
                if past_φ <= 1:
                    # print('over 1, under= 1')
                    φ.append(((-2*past_φ)+1+a) / 1-a)
                # else:
                    # print('phi greater than 1')
            # else:
                # print('phi smaller -1')
            final_len = len(φ)
            if len_phi == final_len:
                
                print( ' nobody added')
                print(past_φ)
        return φ
    
    def set_skew(self, a, duration):
        self.phi_map = self.skew_map(a=a, duration=duration)





    # def split_data(self, train_frac = 0.5):
    #     train_count = int(train_frac * self.size)
    #     print('size',self.size)
    #     print(train_count)
    #     print('len of x',len(self.X))
    #     self.train_x = self.X[0:train_count]
    #     self.validate_x = self.X[train_count+1:]
    #     self.train_y = self.Outputs[0:train_count]
    #     self.validate_y = self.Outputs[train_count+1:]


if __name__ == "__main__":
   sample_dataset = two_spirals() 
   sample_dataset.set_spirals()
   sample_dataset.string_toscaler()
   print(sample_dataset.outputs[0:5])
   print(sample_dataset.y[0:5])
#    sample_dataset.plot_spirals()
    # sample = Chaos_time()
    # sample.set_skew(a=(0.2),duration=5)
    # plt.plot(sample.phi_map)
    # plt.show()
#    print(len(sample_dataset.x))
#    print(sample_dataset.x)
#    print('output',sample_dataset.y)
#    sample_dataset.split_data() they literally didn't use a train test split
#    print('train ys', sample_dataset.train_y)
#    print('validate ys',sample_dataset.validate_y)
# #    print('inputs', sample_dataset.X)
#    print('outputs',sample_dataset.Outputs)
    


    
