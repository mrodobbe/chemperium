from keras.layers import Layer, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU, Concatenate
from keras.regularizers import l2
import tensorflow as tf
from chemperium.inp import InputArguments
# mypy: allow-untyped-defs
# mypy: allow-untyped-calls


class BondInputFeatures(Layer):  # type: ignore[misc]
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.bond_dim = input_shape[0][-1]

        self.W_ib = self.add_weight(
            shape=(self.bond_dim, self.hidden_size),
            initializer="glorot_uniform",
            name="W_ib",
            trainable=True
        )
        self.b_ib = self.add_weight(
            shape=(self.hidden_size),
            initializer="zeros",
            name="b_ib",
            trainable=True
        )

        self.built = True

    def call(self, inputs):
        bond_representations = inputs
        self.bond_inputs = tf.matmul(bond_representations, self.W_ib) + self.b_ib
        self.bond_inputs = tf.nn.relu(self.bond_inputs)

        return self.bond_inputs


def get_distance_weight(xyz, bond_pairs):
    dxyz = tf.gather(xyz, axis=1, indices=bond_pairs[:, :], batch_dims=1)
    xyz1 = dxyz[:, :, 0, 0] - dxyz[:, :, 1, 0]
    xyz1 = xyz1 ** 2
    xyz2 = dxyz[:, :, 0, 1] - dxyz[:, :, 1, 1]
    xyz2 = xyz2 ** 2
    xyz3 = dxyz[:, :, 0, 2] - dxyz[:, :, 1, 2]
    xyz3 = xyz3 ** 2
    xyzd = xyz1 + xyz2 + xyz3
    xyzd = xyzd ** (-1)
    xyzd = tf.where(tf.math.is_inf(xyzd), tf.zeros_like(xyzd), xyzd)
    return xyzd


class DirectedEdgeMessage(Layer):  # type: ignore[misc]
    def __init__(self, hidden_size: int = 64, include_3d: bool = True, mean_readout: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.include_3d = include_3d
        self.mean_readout = mean_readout

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "include_3d": self.include_3d,
            "mean_readout": self.mean_readout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.input_size = input_shape[0][-1]
        self.built = True

    def call(self, inputs):
        bond_representations, bond_pairs, bond_neighbors, xyz = inputs
        distances = get_distance_weight(xyz, bond_pairs)

        if self.include_3d:
            weighted_bond_r = tf.multiply(tf.expand_dims(distances, axis=0)[:, :, :, tf.newaxis], bond_representations)
        else:
            weighted_bond_r = bond_representations

        if self.mean_readout:
            bond_neighbor_features = tf.reduce_mean(tf.gather(weighted_bond_r[0], axis=1,
                                                              indices=bond_neighbors[:, :],
                                                              batch_dims=1), axis=2)
        else:
            bond_neighbor_features = tf.reduce_sum(tf.gather(weighted_bond_r[0], axis=1,
                                                             indices=bond_neighbors[:, :],
                                                             batch_dims=1), axis=2)
        bond_neighbor_features = tf.expand_dims(bond_neighbor_features, axis=0)
        message = bond_neighbor_features

        return message


class MessagePassing(Layer):  # type: ignore[misc]
    def __init__(self, hidden_size: int = 64, depth: int = 4, include_3d: bool = True, mean_readout: bool = False, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.include_3d = include_3d
        self.mean_readout = mean_readout

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "include_3d": self.include_3d,
            "mean_readout": self.mean_readout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.bond_input_step = BondInputFeatures(self.hidden_size)
        self.edge_message_step = DirectedEdgeMessage(self.hidden_size,
                                                     self.include_3d, mean_readout=self.mean_readout)

        self.W_m = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="glorot_uniform",
            name="W_m",
            trainable=True
        )
        self.b_m = self.add_weight(
            shape=(self.hidden_size),
            initializer="zeros",
            name="b_m",
            trainable=True
        )

        self.W_hm = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer="glorot_uniform",
            name="W_hm",
            trainable=True
        )
        self.b_hm = self.add_weight(
            shape=(self.hidden_size),
            initializer="zeros",
            name="b_hm",
            trainable=True
        )

        self.built = True

    def call(self, inputs):
        bond_features, bond_pairs, bond_neighbors, atom_neighbors, xyz = inputs

        bond_representations = self.bond_input_step([bond_features])

        for i in range(self.depth):
            bond_message = self.edge_message_step([bond_representations, bond_pairs, bond_neighbors, xyz])
            bond_message_x = tf.matmul(bond_message, self.W_m) + self.b_m
            bond_message_x = tf.nn.relu(bond_message_x)

            updated_bond = bond_representations + bond_message_x
            bond_representations = tf.nn.relu(tf.matmul(updated_bond, self.W_hm) + self.b_hm)

        return bond_representations


class Readout(Layer):  # type: ignore[misc]
    def __init__(self, hidden_size: int = 256, mean_readout: bool = False, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__()
        self.hidden_size = hidden_size
        self.mean_readout = mean_readout

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "mean_readout": self.mean_readout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        # Make weight matrix for output. input: dim_bonds + dim_atoms, output: hidden_size
        # Make bias vector for output

        self.bond_dim = input_shape[0][-1]
        self.atom_dim = input_shape[1][-1]
        self.cat_dim = self.bond_dim + self.atom_dim

        self.W_o = self.add_weight(
            shape=(self.cat_dim, self.hidden_size),
            initializer="glorot_uniform",
            name="W_o",
            trainable=True
        )
        self.b_o = self.add_weight(
            shape=(self.hidden_size),
            initializer="zeros",
            name="b_o",
            trainable=True
        )
        self.built = True

    def call(self, inputs):

        bond_representations, atomic_features, atom_bond_neighbors = inputs
        atom_neighbor_features = tf.gather(bond_representations[0], atom_bond_neighbors[:, :], axis=1, batch_dims=1)
        atomic_messages = tf.reduce_sum(atom_neighbor_features, axis=2)
        cat_atomic_messages = tf.concat([atomic_features, atomic_messages], axis=2)

        h_atoms = tf.matmul(cat_atomic_messages, self.W_o) + self.b_o
        h_atoms = tf.nn.relu(h_atoms)

        if self.mean_readout:
            h_molecule = tf.reduce_mean(h_atoms, axis=1)
        else:
            h_molecule = tf.reduce_sum(h_atoms, axis=1)

        return h_molecule


def MPNN(
        d_atoms: int,
        d_bonds: int,
        d_out: int,
        hidden_message: int = 64,
        depth: int = 4,
        representation_size: int = 256,
        hidden_size: int = 512,
        layers: int = 5,
        include_3d: bool = True,
        mean_readout: bool = False,
        mfd: bool = False,
        mfd_size: int = 64,
        seed: int = 210995,
        activation: str = "linear",
        dropout: float = 0.0,
        batch_normalization: bool = False,
        l2_value: float = 0.0
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    atom_features = Input(shape=[100, d_atoms], dtype="float32", name="initial_atom_features")
    bond_features = Input(shape=[250, d_bonds], dtype="float32", name="initial_bond_features")
    molecular_features = Input(shape=[1, 13], dtype="float32", name="initial_mol_features")
    bond_pairs = Input(shape=[250, 2], dtype="int32", name="bond_pairs")
    xyz = Input(shape=[100, 3], dtype="float32", name="coordinates")

    bond_neighbors = Input(shape=[250, 8], dtype=tf.int64, name="bond_neighbors")
    atom_neighbors = Input(shape=[100, 8], dtype=tf.int64, name="atom_neighbors")
    atom_bond_neighbors = Input(shape=[100, 8], dtype=tf.int64, name="atom_bond_neighbors")

    bonds_t = MessagePassing(hidden_message, depth, include_3d, mean_readout)(
        [bond_features,
         bond_pairs,
         bond_neighbors,
         atom_neighbors,
         xyz]
        )

    x = Readout(representation_size)([bonds_t, atom_features, atom_bond_neighbors])
    if mfd:
        mx = Dense(mfd_size)(molecular_features)
        mx = LeakyReLU()(mx)
        mx = tf.squeeze(mx, [1])
        x = Concatenate()([x, mx])

    for layer in range(layers):
        x = Dense(hidden_size, kernel_regularizer=l2(l2_value))(x)
        x = BatchNormalization()(x) if batch_normalization else x  # Batch Normalization
        x = LeakyReLU()(x)
        x = Dropout(dropout)(x) if dropout > 0 else x  # Dropout Layer

    x = Dense(d_out, activation=activation)(x)

    model = Model(inputs=[atom_features, bond_features, bond_pairs, xyz,
                          bond_neighbors, atom_neighbors, atom_bond_neighbors, molecular_features], outputs=[x])

    return model
