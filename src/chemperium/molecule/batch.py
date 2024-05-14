def featurize_graphs(mol_graphs):
    all_atom_features = []
    all_bond_features = []
    all_bond_pairs = []
    xyz = []
    bond_neighbors = []
    atom_neighbors = []
    atom_bond_neighbors = []
    all_mol_features = []

    for graph in mol_graphs:
        all_atom_features.append(graph.atom_features)
        all_bond_features.append(graph.bond_representations)
        all_bond_pairs.append(graph.bond_pairs)
        xyz.append(graph.xyz)
        bond_neighbors.append(graph.bond_neighbors)
        atom_neighbors.append(graph.atom_neighbors)
        atom_bond_neighbors.append(graph.atom_bond_neighbors)
        all_mol_features.append(graph.mol_features)

    all_atom_features = tf.ragged.constant(all_atom_features, dtype=tf.float32)
    all_bond_features = tf.ragged.constant(all_bond_features, dtype=tf.float32)
    all_bond_pairs = tf.ragged.constant(all_bond_pairs, dtype=tf.int64)
    xyz = tf.ragged.constant(xyz, dtype=tf.float32)
    all_bond_neighbors = tf.ragged.constant(bond_neighbors, dtype=tf.float32)
    atom_neighbors = tf.ragged.constant(atom_neighbors, dtype=tf.int64)
    atom_bond_neighbors = tf.ragged.constant(atom_bond_neighbors, dtype=tf.int64)
    all_mol_features = tf.ragged.constant(all_mol_features, dtype=tf.float32)

    return (all_atom_features, all_bond_features, all_bond_pairs,
            xyz, all_bond_neighbors, atom_neighbors, atom_bond_neighbors,
            all_mol_features)


def prepare_batch(x_batch, y_batch):
    all_atom_features, all_bond_features, all_bond_pairs, xyz, all_bond_neighbors, \
        all_atom_neighbors, all_atom_bond_neighbors, all_mol_features = x_batch

    all_bond_pairs = all_bond_pairs.to_tensor(99, shape=[None, 250, 2])

    all_atom_features = all_atom_features.to_tensor(shape=[None, 100, all_atom_features[0][0].shape[0]])
    all_bond_features = all_bond_features.to_tensor(shape=[None, 250, all_bond_features[0][0].shape[0]])
    all_mol_features = all_mol_features.to_tensor(shape=[None, 1, 13])

    xyz = xyz.to_tensor(shape=[None, 100, 3])

    all_bond_neighbors = tf.cast(all_bond_neighbors.to_tensor(249, shape=[None, 250, 8]), tf.int32)

    all_atom_neighbors = all_atom_neighbors.to_tensor(99, shape=[None, 100, 8])
    all_atom_bond_neighbors = all_atom_bond_neighbors.to_tensor(249, shape=[None, 100, 8])

    return (all_atom_features, all_bond_features, all_bond_pairs, xyz,
            all_bond_neighbors, all_atom_neighbors, all_atom_bond_neighbors, all_mol_features), y_batch


def MPNNDataset(x, y, batch_size=32, shuffle=False, seed=210995):
    dataset = tf.data.Dataset.from_tensor_slices((x, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024, seed=seed)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
