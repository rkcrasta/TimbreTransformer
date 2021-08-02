import numpy as np
import random


def rescale(array, scaling_factor = 1e4, threshold = 1):
    """
    Divide array by it's maximum to make all inputs uniform. 
    The normalized arrays are then multiplied by a large scaling factor to make it easier for the network to train on them

    Function returns rescaled array along with the original array_maximum which is used to return the scaled array back to the way it was
    """

    scaled = np.add(array, threshold)

    scaled = np.divide(scaled, threshold)

    scaled = np.log(scaled)

    array_max = np.amax(np.abs(scaled))

    #Prevent divide by zero. array_max == 0 implies that the whole array is 0, so the replacement value is arbitrary
    if array_max == 0:
        array_max = 1
    
    scaled = np.divide(scaled, array_max)

    rescaled = np.multiply(scaled, scaling_factor)

    return rescaled, array_max

def undo_rescale(array, original_array_max, scaling_factor=1e4, threshold=1):
    """
    Undoes the effects of rescale
    """
    rescaled = np.divide(array, scaling_factor)

    original = np.multiply(rescaled, original_array_max)

    original = np.exp(original)

    original = np.multiply(original, threshold)

    original = np.subtract(original, threshold)

    return original


def partition(audio_array, randomize = False, input_shape = (129,500,1), batch_size = 200):
    """
    Splits the audio array into the correct size for input to the network

    randomize == True is used when training the network 
    
    randomize == False is used when making predictions
    """

    #This will be returned as an np.array
    output = []
    scale_list = []

    #Zero pad array on the right
    padded = np.pad(audio_array, ((0,0),(0,int(input_shape[1]))))
    if randomize:

        for i in range(batch_size):

            start = random.randint(0,audio_array.shape[1])

            rescaled, scale = rescale(padded[:,start:start+input_shape[1]])

            output.append(rescaled)
            scale_list.append(scale)
    
    else:
    #
        start = 0
        while start <= audio_array.shape[1]:

            rescaled, scale = rescale(padded[:,start:start+input_shape[1]])

            output.append(rescaled)

            scale_list.append(scale)

            start += input_shape[1]

    return (np.array(output), np.array(scale_list))


def combine(partitioned_array, scale_list):
    """
    Reverses the operation of the combine function
    """
    output = []

    for i in range(partitioned_array.shape[0]):
        
        output.append(undo_rescale(partitioned_array[i],scale_list[i]))

    return np.concatenate(output, axis = 1)






