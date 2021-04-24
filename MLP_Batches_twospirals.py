# MLP_Batches_twospirals.py



import numpy as np
import matplotlib.pyplot as plt
from Homemade_datasets import Nparity_dataset, two_spirals
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
    bias= -1.0
    return nonlin((np.dot(inputs,synapse)+ bias))

def think_astro(inputs, synapse):
    bias= -1.0
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
dataset = two_spirals(size=35)
dataset.set_spirals()
dataset.string_toscaler()
# dataset.plot_spirals()
dataset.test_train_split(0.8)

# input_x = dataset.x
input_x = dataset.train_x
validate_input = dataset.test_x

# output_y = dataset.y
output_y = dataset.train_y
validate_output = dataset.test_y

print('len input',len(input_x))
print('len validate input',len(validate_input))
#network parameters
hidden_layer_count = 1 #needs at least 1 hidden unit
hidden_units = 30 #all hidden layers have the same amount
output_units = len(output_y[0])
total_layer_count = hidden_layer_count + 2
epoch_count = 5000
l_rate = 0.1

vale_plot = []
vale_sse = 0
compute_validation = True
batch_number = 100
current_batch = 0
batches = []
# final_batches = []
#special parameters
astro_status = False
backpropastro = False
if astro_status == True:

    # start_vals = np.random.random(3)
    ## random
    # start_vals = list(np.random.random(2))
    # start_vals.append(1.0)

    #set manually
    # start_vals = [0.01,0.1,0.11] #[decay, threshold, weight]
    # start_vals[0] = np.random.random() #between 0 and 1
    # start_vals = [0.5,0.1,-0.1] #[decay, threshold, weight] their values
    start_vals = [0.66, 0.28, 1.0] #my values
    # start_vals[2] = np.random.random() #between 0 and 1


    backpropastro = False #follows backpropogation 
    train_decay = False #trained by setting value to inverse of average activity of corresponding astro (each individually)
    train_threshold = False #trained by setting value to running average of corresponding astro activity (each have their own)

    anne = AAN(size=(hidden_layer_count, hidden_units), decay_rate=start_vals[0], threshold=start_vals[1],weight=start_vals[2],backprop_status=backpropastro)
    anne.set_parameters()

    if backpropastro == True:
        astro_l_rate = l_rate
        
    




while current_batch < batch_number:

    syn_list = []
    syn01 = 2*np.random.random(((len(input_x[0])),hidden_units)) -1
    syn_list.append(syn01)

    for layer in range(0,hidden_layer_count-1):

        syn_hid = 2*np.random.random((hidden_units,hidden_units)) -1
        syn_list.append(syn_hid)


    synoutput = 2* np.random.random((hidden_units,output_units))-1  # 2* -1 to center random values around 0
    syn_list.append(synoutput)



    print('beginning testing, Epoch = 0/'+str(epoch_count))
    # SSE_Plot = []
    for i in range(epoch_count):
        SSE_Plot = []
        epoch_error = np.zeros(output_units)
        epoch_error = epoch_error.reshape(1,output_units)
        sse_perepoch = 0


        for train_unit_count in range(0, int(len(input_x))-1):

            current_sse = 0

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
            # print('net error',net_error)
            epoch_error += net_error
            sse_perepoch += SSE(layer_list[-1],output_y[train_unit_count])
            current_sse = SSE(layer_list[-1],output_y[train_unit_count])

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

                if backpropastro == True:
                    if synapse_count < start_synapse_count:
                        astro_adjust = new_layer_error * anne.activity[synapse_count]                    
                        anne.weights[synapse_count] += astro_adjust * l_rate

        

        SSE_Plot.append(sse_perepoch / train_unit_count) #find running average SSE
        if i%100 == 0:
            print('current Epoch:',i)
            print('average_sse', sse_perepoch / train_unit_count)

    # final_sse = SSE(layer_list[-1],output_y[train_unit_count])
    batches.append(SSE_Plot[-1])
    # batches.append(final_sse)
    ### compute validation error
    if compute_validation == True:
        # if train_unit_count % 50 == True:
        vale_sse = 0
        val_sse_perepoch = 0
        ##### validation accuracy with test dataset
        for test_unit_count in range(0, int(len(validate_input))-1):  
            train_unit = validate_input[test_unit_count]
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
            # val_net_error = Error(layer_list[-1],validate_output[test_unit_count])
            # print('val net error',val_net_error)
            # print('leyer list -1',layer_list[-1])
            # print('validate output',validate_output[test_unit_count])
            # epoch_error += val_net_error
            val_sse_perepoch += SSE(layer_list[-1],validate_output[test_unit_count])

        vale_sse = val_sse_perepoch / test_unit_count
        # np.average()
        vale_plot.append(vale_sse)
    

    # final_batches.append(SSE(layer_list[-1],output_y[train_unit_count]))
    current_batch +=1


average = np.sum(batches) / len(batches)
max_sse = 1
for sse in batches:
    if sse < max_sse:
        max_sse = sse
std = np.std(batches)
# print('n',n)
print('final average sse', average)
print('final sse',batches[-1])
print('standard deviation', std)
print('best sse', max_sse)
print('no astro, no bias')

print('validation')
print('average validate sse',np.average(vale_plot))
# print('vale plot length', len(vale_plot)) 
# print('vale plot',vale_plot)
print('vale std',np.std(vale_plot))
print('none')


