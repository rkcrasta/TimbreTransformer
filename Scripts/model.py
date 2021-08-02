import tensorflow as tf
import custom_loss
import json
import os

def create_model(name, n_fft, srate, input_shape_y, custom_loss_parameter, save_dir, discriminator, generator):
    
    with open('model_params.txt', 'r+') as params_file:
        params = json.load(params_file)

    model_params = {"save_dir": save_dir, "n_fft":n_fft, "srate":srate, "input_shape_y":input_shape_y, "custom_loss_parameter":custom_loss_parameter}
    params[name] = model_params

    with open('model_params.txt', 'r+') as params_file:
        json.dump(params, params_file, indent = 4)
    
    input_shape = (int((n_fft/2)+1), int(input_shape_y), 1)
    
    disc = discriminator(input_shape)
    gen = generator(input_shape)
    gan_input_layer = gen.input
    gan_output = disc(gen.layers[-1].output)
    
    model = tf.keras.Model(inputs = gan_input_layer, outputs = [gan_output, gen.layers[-1].output])

    tf.keras.models.save_model(model, os.path.join(save_dir,name))

class Model():
    def __init__(self, model_name):
        
        self.model_params = json.load(open('model_params.txt'))[model_name]
        self.input_shape = (int((self.model_params['n_fft']/2)+1), int(self.model_params['input_shape_y']), 1)
        self.model_name = model_name

    def load_from_file(self):

        self.model= tf.keras.models.load_model(os.path.join(self.model_params['save_dir'],self.model_name), custom_objects = {'custom_loss': custom_loss.custom_loss(self.model_params['custom_loss_parameter'])}, compile = False)
        
        
        #Create generator from the generator layers of the saved model
        self.generator = tf.keras.Model(inputs = self.model.input, outputs = self.model.layers[-2].output)

        self.generator.compile(loss=custom_loss.custom_loss(self.model_params['custom_loss_parameter']), optimizer = 'adam')

        #Create discriminator from the last layer of the saved model
        self.discriminator = tf.keras.Model(inputs = self.model.layers[-1].input, outputs = self.model.layers[-1].output)

        #trainable is set to True before compiling to allow it to train when it's fit function is called
        self.discriminator.trainable = True
        self.discriminator.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = 'adam', metrics=['BinaryAccuracy'])
        #trainable is set to False after compiling to prevent it from being trained when the generator is being trained
        self.discriminator.trainable = False

        self.model.compile(loss = [tf.keras.losses.BinaryCrossentropy(), custom_loss.custom_loss(self.model_params['custom_loss_parameter'])], optimizer = 'adam', metrics={'discriminator':'BinaryAccuracy'})
        return self
        

