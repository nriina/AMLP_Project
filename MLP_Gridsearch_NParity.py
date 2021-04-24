#File for performing grid searches with AMLP on the homemade nparity file

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
# n = 5
# dataset = Nparity_dataset(N= n)
# dataset.populate()

# input_x = dataset.X

# output_y = dataset.Outputs

########################################### 2 spirals load dataset
################################### load dataset spirals
dataset = two_spirals(size=100)
dataset.set_spirals()
dataset.string_toscaler()
# dataset.plot_spirals()
dataset.test_train_split(0.8)

# input_x = dataset.x
input_x = dataset.train_x

# output_y = dataset.y
output_y = dataset.train_y

#################################3


#network parameters
hidden_layer_count = 1 #needs at least 1 hidden unit
hidden_units = n #all hidden layers have the same amount
output_units = len(output_y[0])
total_layer_count = hidden_layer_count + 2
epoch_count = 500
l_rate = 0.1

#best combo: [0.1, 0.11, 0.01], best sse 0.9346685300687123

weight_iterations = [-1.00, -0.78, -0.56,-0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1.00]
decay_iterations = [0.01, 0.12, 0.23, 0.34, 0.45, 0.55, 0.66, 0.77, 0.88, 0.99]
thresh_iterations = [0.10, 0.19, 0.28, 0.37, 0.46, 0.54, 0.63, 0.72, 0.81, 0.90]

iterations = 5

vegetables_list = []
farmers_list = []
harvest_list = []

#special parameters
astro_status = True
if astro_status == True:
    weight_graphs = []
    for weight in weight_iterations:
        # weight_test = weight_iterations[0]
        decay_list = []
        for decay in decay_iterations:
            
            thresh_list = []
            for thre in thresh_iterations:
                
                for it in range(iterations):
                    it_list = []
                    # start_vals = np.random.random(3)
                    start_vals = [decay,thre,weight] #[decay, threshold, weight]



                    backpropastro = False
                    train_decay = True
                    train_threshold = False

                    anne = AAN(size=(hidden_layer_count, hidden_units), decay_rate=start_vals[0], threshold=start_vals[1],weight=start_vals[2],backprop_status=backpropastro)
                    anne.set_parameters()

                    if backpropastro == True:
                        astro_l_rate = l_rate
        
                    

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

                                    if synapse_count < start_synapse_count:
                                        astro_adjust = new_layer_error * anne.activity[synapse_count]                    
                                        anne.weights[synapse_count] += astro_adjust * l_rate

                        

                        SSE_Plot.append(sse_perepoch / train_unit_count) #find running average SSE
                        if i%100 == 0:
                            print('current Epoch:',i)
                            print('average_sse', sse_perepoch / train_unit_count)


                    ##plot that bitch's fitness over time
                    final_sse = SSE_Plot[-1]
                    it_list.append(final_sse)
                    # plt.plot(SSE_Plot)
                    # plt.show()
                average_sse = 0
                for ss in range(len(it_list)):
                    average_sse += it_list[ss]
                    average_sse = average_sse / len(it_list)
                # thresh_list.append(average_sse)
                thresh_list.append( 1 - average_sse) #change to accuracy
                # print('threshlist',thresh_list)
            decay_list.append(thresh_list)

        weight_graphs.append(np.array(decay_list))


    ### plot code from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]

    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

# for weight_name in range(len(weight_iterations)):
#     
#     # harvest = np.array(decay_list)
#     harvest = weight_graphs[weight_name]
#     # weight_graphs.append(harvest)
#     print('harvest',harvest)
#     # print('threshlist',thresh_list)
#         # print('decay list',decay_list)


    # fig, ax = plt.subplots(len(weight_iterations), 1, weight_name+1)

fig, axs = plt.subplots(1,len(weight_iterations))

# vegetables = decay_iterations
# farmers = thresh_iterations
vegetables = []
farmers = []

print('nparity redo mini')
for weight_g in range(len(weight_graphs)):
    # print('weight graph',weight_graphs[weight_g])
    axs[weight_g].imshow(weight_graphs[weight_g])
    axs[weight_g].set_title('Weight: '+str(weight_iterations[weight_g]))

    # We want to show all ticks...
    axs[weight_g].set_xticks(np.arange(len(farmers)))
    axs[weight_g].set_yticks(np.arange(len(vegetables)))
    # # ... and label them with the respective list entries
    # axs[weight_g].set_xticklabels(farmers)
    # axs[weight_g].set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    # plt.setp(axs[weight_g].get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    
plt.show()

best = 0
best_it = [0,0,0]
for ite in range(len(weight_graphs)):
    for decl in range(len(weight_graphs[ite])):
        for wait in range(len(weight_graphs[ite][decl])):
            # print('wait',wait)
            # print('wait entry',weight_graphs[ite][decl][wait])
            if weight_graphs[ite][decl][wait] > best:
                best = weight_graphs[ite][decl][wait]
                best_it[0] = wait #which thresh
                best_it[1] = decl #which decay
                best_it[2] = ite #which number of wait iteration

print('best sse for two spirals',best)
print('best combo thresh:',  thresh_iterations[best_it[0]])
print('best combo1 decl:',  decay_iterations[best_it[1]])
print('best combo2 wait:',weight_iterations[best_it[2]])


# im = axs[0].imshow(harvest)
# # We want to show all ticks...
# im.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(farmers)
# ax.set_yticklabels(vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     # for i in range(len(vegetables)):
#     #     for j in range(len(farmers)):
#     #         text = ax.text(j, i, harvest[i, j],
#     #                        ha="center", va="center", color="w")

# ax.set_title("gridsearch for weight" + str(weight_iterations[weight_name]))
# fig.tight_layout()
# plt.show()



    # fig, ax = plt.subplots(5, 1, 1)
    # im = ax.imshow(harvest)

    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(farmers)))
    # ax.set_yticks(np.arange(len(vegetables)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # # for i in range(len(vegetables)):
    # #     for j in range(len(farmers)):
    # #         text = ax.text(j, i, harvest[i, j],
    # #                        ha="center", va="center", color="w")

    # ax.set_title("gridsearch for weight" + str(weight))
    # fig.tight_layout()
    # # plt.show()





# #trial
# trial_num = 0
# trial_output = output_y[trial_num]
# print('target', trial_output)

# #run value back through network
# train_unit = input_x[trial_num]
# train_unit = np.asarray(train_unit)
# layer_list = []

# new_unit = Flatt(train_unit)
# new_unit = np.asarray(new_unit)


# layer_list.append(new_unit.T)

# for lay in range(1,total_layer_count):
#     prev_layer = layer_list[lay-1]
#     layer = think(prev_layer,syn_list[lay-1])
#     layer_list.append(layer)


# network_output = layer_list[-1]
# print('actual',network_output)

