from keras import layers, activations, initializers, Sequential, Input
from keras.regularizers import l2
from chemperium.inp import InputArguments


class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int, inp: InputArguments):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = inp.hidden_size
        self.batch_size = inp.batch_size
        self.num_layers = inp.num_layers
        self.seed = inp.seed

        model = Sequential()
        model.add(Input(shape=(self.input_size,), name="input"))
        if self.num_layers == 1:
            model.add(layers.Dense(self.hidden_size, kernel_initializer=self.initializer(self.seed),
                                   bias_initializer=self.initializer(self.seed), name="layer_1"))
        else:
            for i in range(self.num_layers - 1):
                model.add(layers.Dense(self.hidden_size, kernel_initializer=self.initializer(self.seed + i),
                                       bias_initializer=self.initializer(self.seed + i),
                                       name=str("layer_" + str(i + 1))))
                if inp.batch_normalization:
                    model.add(layers.BatchNormalization())  # Batch Normalization
                model.add(self.get_activation_function(inp.hidden_activation, i+1))
                if inp.dropout > 0:
                    model.add(layers.Dropout(rate=inp.dropout))

            model.add(layers.Dense(self.hidden_size,
                                   kernel_initializer=self.initializer(self.seed + self.num_layers),
                                   bias_initializer=self.initializer(self.seed + self.num_layers),
                                   kernel_regularizer=l2(inp.l2),
                                   name=str("layer_" + str(self.num_layers))))

        model.add(layers.Dense(self.output_size,
                               activation=inp.activation,
                               kernel_initializer=initializers.RandomUniform(seed=self.seed),
                               bias_initializer=initializers.RandomUniform(seed=self.seed),
                               name="output"))

        # model = keras.Sequential(nn_layers)

        model.summary()
        model.compile(loss='mean_squared_error',
                      optimizer="Adam",
                      metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

        self.model = model

    def get_activation_function(self, activation, layer: int):
        layer = str(layer)
        if activation == 'ReLU':
            return layers.Activation(activations.relu, name="ReLU_" + layer)
        elif activation == 'LeakyReLU':
            return layers.LeakyReLU(name="LeakyReLU_" + layer)
        elif activation == 'swish':
            return layers.Activation(activations.swish, name="swish_" + layer)
        elif activation == 'tanh':
            return layers.Activation(activations.tanh, name="tanh_" + layer)
        elif activation == 'SELU':
            return layers.Activation(activations.selu, name="selu_" + layer)
        elif activation == 'ELU':
            return layers.Activation(activations.elu, name="elu_" + layer)
        elif activation == "sigmoid":
            return layers.Activation(activations.sigmoid, name="sigmoid_" + layer)
        else:
            raise ValueError(f'Activation "{activation}" not supported.')

    def initializer(self, seed):
        return initializers.glorot_uniform(seed=seed)
