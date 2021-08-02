# TimbreTransformer

## Description

A project to train a neural network to transform the timbre of an audio file and make it sound like the training data. 

The ipython notebooks are designed to be used with Google Colab, so you will need a Google account to use them as is.

## Use

### For Each Notebook:

 - Change the SCRIPTS_PATH to the absolute path to the Scripts folder
 - Change the DEFAULT_PATH to the absolute path to working directory where the notebooks are located

### Create_Model

 - Set the MODEL_PARAMS to the desired model parameters

 - Define the discriminator and generator models. 
 	- The discriminator should return the probability that the input is from the training data
 	- The generator is an autoencoder that returns an output of the same shape as the input

### Train_Model

 - TRAIN_PARAMS:
	- 'model_name': Name of the model that will be trained. It should be a key in the 'model_parameters.txt' file
	- 'target_audio': A list of the audio files that the model will train to sound like
	- 'train_audio': A list of the audio files that the model will train to transform
 - Set the n_epochs and n_batches that the model will train for

### Transform_Audio
 - takes an input audio file, transforms it and saves the result as a wav file.
 - PARAMS:
	- 'model_name': Name of the model that will transform the input_audio
	- 'input_audio': The audio file that will be transformed
	- 'output_audio': The transformed file

### Scripts

 - custom_loss: The loss function that returns the similarity between the model input and output. This is to ensure that the model output has the same content as the model input

 - import_audio: Defines class Audio and class FourierTransform and is used to load the audio files and convert them to arrays

 - model: Defines class Model that is used to save and load the model and it's parameters across sessions

 - process_audio: defines functions that are used to prepare the model input
