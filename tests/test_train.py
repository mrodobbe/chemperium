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

    cp.train(example_file, prop, save_dir, input_arguments)
    print("Training test finished successfully!")
