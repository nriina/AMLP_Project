# this is a file for creating the N-parity dataset to train a mlp

import numpy as np
import math
import matplotlib.pyplot as plt
import random

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
        print('len of x',len(self.X))   
    

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
        pass

    def spiral_xy(self, i, spiral_num): # mathcode from https://conx.readthedocs.io/en/latest/Two-Spirals.html
        """
        Create the data for a spiral.

        Arguments:
            i runs from 0 to 96
            spiral_num is 1 or -1
        """
        φ = i/16 * math.pi
        r = 6.5 * ((104 - i)/104) #104, lower this value to make spirals less tight, discovered after presentation
        x = (r * math.cos(φ) * spiral_num)/13 + 0.5
        y = (r * math.sin(φ) * spiral_num)/13 + 0.5
        return (x, y)

    def spiral(self,spiral_num):
        return [self.spiral_xy(i, spiral_num) for i in range(self.range)] #range is the amount of points 

    def set_spirals(self):
        self.a = ["A", self.spiral(1)]
        self.b = ["B", self.spiral(-1)]
        total_list = self.a[1] + self.b[1]
        self.x = random.sample(total_list, len(total_list))
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
                print('unknown value')
        self.y = datapoint
        return datapoint


    def plot_spirals(self):
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
        self.test_y = self.y[train_length:]
            


#didn't end up using this, on a different paper

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
            output = self.calc_output(instance)
            self.y.append(output)




#didn't end up using, for a different paper
class Chaos_time():
    def __init__(self):
        self.phi_map = []
        pass
    def skew_map(self,a, duration, initial_val=0.43):
        # initial_value = initial_val
        φ = [initial_val] #holds y axis of time series
        for i in range(0,duration):
            past_φ = φ[i]
            len_phi = len(φ)
            if past_φ > 1:
                past_φ = 0.99 #stuck at -1
            if past_φ == -1:
                past_φ = -0.99

            if -1 <= past_φ:
                if past_φ <= a:
                    φ.append(((2*past_φ)+1-a) / 1+a)
            if a < past_φ:
                if past_φ <= 1:
                    φ.append(((-2*past_φ)+1+a) / 1-a)
            final_len = len(φ)
            if len_phi == final_len:
                
                print( ' nobody added')
                print(past_φ)
        return φ
    
    def set_skew(self, a, duration):
        self.phi_map = self.skew_map(a=a, duration=duration)


if __name__ == "__main__":
   sample_dataset = two_spirals() 
   sample_dataset.set_spirals()
   sample_dataset.string_toscaler()
   sample_dataset.plot_spirals()
    # sample = Chaos_time()
    # sample.set_skew(a=(0.2),duration=5)
    # plt.plot(sample.phi_map)
    # plt.show()
#    print(len(sample_dataset.x))
#    print(sample_dataset.x)
#    print('output',sample_dataset.y)
#    sample_dataset.split_data() they literally didn't use a train test split

    


    
