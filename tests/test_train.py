import chemperium as cp


def run_training():
    example_file = "../test_data/test_example.csv"
    prop = "magic_property"
    save_dir = "../test_data/test_output"
    input_arguments = {"batch": 16,
                       "seed": 123456789,
                       "epochs": 2,
                       "patience": 1,
                       "depth": 4,
                       "folds": 1,
                       "num_layers": 2}

    cp.training.train.train(example_file, prop, save_dir, input_arguments)
    print("Training test finished successfully!")


def run_training_fp():
    example_file = "../test_data/test_example.csv"
    prop = "magic_property"
    save_dir = "../test_data/test_output"
    input_arguments = {"batch": 16,
                       "seed": 123456789,
                       "epochs": 2,
                       "patience": 1,
                       "depth": 4,
                       "folds": 1,
                       "num_layers": 2,
                       "fingerprint": "morgan"}

    cp.training.train.train(example_file, prop, save_dir, input_arguments)
    print("Training test finished successfully!")


def run_training_gaul():
    example_file = "../test_data/test_example.csv"
    prop = "magic_property"
    save_dir = "../test_data/test_output"
    input_arguments = {"batch": 16,
                       "seed": 123456789,
                       "epochs": 2,
                       "patience": 1,
                       "depth": 4,
                       "folds": 1,
                       "num_layers": 2,
                       "distances": True,
                       "angles": True,
                       "dihedrals": True,
                       "include_3d": True,
                       "ff_3d": True,
                       "fingerprint": "hdad"}

    cp.training.train.train(example_file, prop, save_dir, input_arguments)
    print("Training test finished successfully!")
