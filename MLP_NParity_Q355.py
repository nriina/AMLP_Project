# MLP_NParity_Project code

import numpy as np
import matplotlib.pyplot as plt
from Homemade_datasets import Nparity_dataset
from AMLP_gergel import AAN

def nonlin(x, derive = False):
    if derive == True:
        return (x *(1 - x))
    else:
        return 1 /(1+(np.exp(-x)))

def SSE(network, actual):
    error = 0
    for i in range(0,len(network),1):
        error += ((actual[i] - network[i])**2)
    bigerror = (np.sum(error))*0.5
    return bigerror

def think(inputs, synapse):
    bias= -1
    return nonlin((np.dot(inputs,synapse)+ bias))

def think_astro(inputs, synapse):
    bias= -1
    return (np.dot(inputs,synapse)+ bias)

def Flatt(inputlist):
    output_list = []
    for row in inputlist:
            output_list.append(row)
    return output_list

def Error(networkoutput, actual): #actual is 0-9, network is (1,10)

    represent = actual
    network_error = actual - networkoutput

    return network_error

################################### load dataset
n = 6
dataset = Nparity_dataset(N= n)
dataset.populate()

input_x = dataset.X

output_y = dataset.Outputs




#network parameters
hidden_layer_count = 1 #needs at least 1 hidden unit
hidden_units = 20 #all hidden layers have the same amount
output_units = len(output_y[0])
total_layer_count = hidden_layer_count + 2
epoch_count = 500
l_rate = 0.1

#special parameters
astro_status = True
if astro_status == True:

    start_vals = np.random.random(3)
    # start_vals = [0.5,0.5,1] #[decay, threshold, weight]
    backpropastro = True
    train_decay = True
    train_threshold = True

    anne = AAN(size=(hidden_layer_count, hidden_units), decay_rate=start_vals[0], threshold=start_vals[1],weight=start_vals[2],backprop_status=backpropastro)
    anne.set_parameters()
    









syn_list = []
syn01 = 2*np.random.random(((len(input_x[0])),hidden_units)) -1
syn_list.append(syn01)

for layer in range(0,hidden_layer_count-1):

    syn_hid = 2*np.random.random((hidden_units,hidden_units)) -1
    syn_list.append(syn_hid)


synoutput = 2* np.random.random((hidden_units,output_units))-1  # 2* -1 to center random values around 0
syn_list.append(synoutput)



print('beginning testing, Epoch = 0/'+str(epoch_count))
SSE_Plot = []
for i in range(epoch_count):
    epoch_error = np.zeros(output_units)
    epoch_error = epoch_error.reshape(1,output_units)
    sse_perepoch = 0


    for train_unit_count in range(0, int(len(input_x))-1):


  
        train_unit = input_x[train_unit_count]
        train_unit = np.asarray(train_unit)
        layer_list = []

        new_unit = Flatt(train_unit)
        new_unit = np.asarray(new_unit)
    

        layer_list.append(new_unit.T)

        for lay in range(1,total_layer_count):
            prev_layer = layer_list[lay-1]

            if astro_status == False:
                layer = think(prev_layer,syn_list[lay-1])
                layer_list.append(layer)

            else:
                layer = think_astro(prev_layer,syn_list[lay-1])

                if lay < (total_layer_count-1): #index is one less than total layer count
                    anne.input[lay-1] = nonlin(layer)   #lay-1 because layer 1 for neuron is layer 0 for glia
                    if train_decay == False:
                        if train_threshold == False: #both false
                            anne.compute_activation()
                        else: #decay false, threshold true
                            anne.compute_activation_theta()

                    elif train_decay == True: 
                        if train_threshold == False: #decay true, threshold false
                            anne.compute_activation_decay()
                        else:
                            anne.compute_activation_all() #all true


                    for n in range(0,len(layer)):
                        layer[n] += (anne.activity[lay-1][n] * anne.weights[lay-1][n])
                layer_list.append(nonlin(layer))


        #compute error
        net_error = Error(layer_list[-1],output_y[train_unit_count])
        epoch_error += net_error
        sse_perepoch += SSE(layer_list[-1],output_y[train_unit_count])

        ####################adjust weights 
        final_delta = net_error * nonlin(layer_list[-1], derive= True)
        delta_list = []
        delta_list.append(np.asarray(final_delta)) #delta is backwards should be 6 total deltas (for every layer except input)

        ######back prop

        synapse_count = len(syn_list)-1
        
        start_synapse_count = len(syn_list)-1
        while synapse_count > 0:

            incoming_delta = delta_list[start_synapse_count - synapse_count]

            syp_inq = syn_list[synapse_count] 

            new_layer_error = incoming_delta.dot(syp_inq.T)

            next_delta = new_layer_error * nonlin(layer_list[synapse_count],derive= True)

            syn_adj = np.dot(layer_list[synapse_count].reshape(len(layer_list[synapse_count]),1),incoming_delta.reshape(len(incoming_delta),1).T)

            syn_list[synapse_count] += syn_adj * l_rate

            delta_list.append(np.asarray(next_delta))

            synapse_count -= 1

            if anne.backprop == True:
                # print('attempting backprop')
                # if synapse_count > 0:
                if synapse_count < start_synapse_count:
                    astro_adjust = new_layer_error * anne.activity[synapse_count]                    
                    anne.weights[synapse_count] += astro_adjust * l_rate

    

    SSE_Plot.append(sse_perepoch / train_unit_count) #find running average SSE
    if i%100 == 0:
        print('current Epoch:',i)
        print('average_sse', sse_perepoch / train_unit_count)


##plot that bitch's fitness over time
plt.plot(SSE_Plot)
plt.show()

#trial
trial_num = np.random.random_integers(low=0, high=train_unit_count)
trial_output = output_y[trial_num]
print('target', trial_output)

#run value back through network
train_unit = input_x[trial_num]
train_unit = np.asarray(train_unit)
layer_list = []

new_unit = Flatt(train_unit)
new_unit = np.asarray(new_unit)


layer_list.append(new_unit.T)

for lay in range(1,total_layer_count):
    prev_layer = layer_list[lay-1]
    layer = think(prev_layer,syn_list[lay-1])
    layer_list.append(layer)


network_output = layer_list[-1]
print('actual',network_output)

