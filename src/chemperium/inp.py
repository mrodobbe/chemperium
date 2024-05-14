import argparse
import os.path as path


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Train the 3D MPNN")
        self.parser.add_argument("--save_dir", type=str, help="Folder to store results")
        self.parser.add_argument("--property", type=str, help="Target property/ies")
        self.parser.add_argument("--data", type=str, help="Choose dataset")

        self.parser.add_argument("--test_data", type=str, help="Choose external test data")
        self.parser.add_argument("--transfer_data", type=str, help="Choose transfer data")
        self.parser.add_argument("--transfer_property", type=str, help="Choose property to be predicted")

        self.parser.add_argument("--batch", type=int, help="Choose batch size", default=128)
        self.parser.add_argument("--transfer_batch", type=int, help="Choose batch size for transfer data", default=128)
        self.parser.add_argument("--seed", type=int, help="Random generator seed", default=210995)
        self.parser.add_argument("--epochs", type=int, help="Maximal number of epochs", default=700)
        self.parser.add_argument("--patience", type=int, help="Choose patience", default=75)
        self.parser.add_argument("--transfer_patience", type=int,
                                 help="Choose patience for transferred model", default=30)
        self.parser.add_argument("--depth", type=int, help="Message-passing radius", default=6)
        self.parser.add_argument("--folds", type=int, help="Number of folds for ensemble", default=10)
        self.parser.add_argument("--num_layers", type=int, help="Choose number of hidden layers", default=5)
        self.parser.add_argument("--hidden_size", type=int,
                                 help="Choose number of neurons per hidden layer", default=500)
        self.parser.add_argument("--hidden_message", type=int,
                                 help="Choose number of neurons per hidden bond representation", default=512)
        self.parser.add_argument("--representation_size", type=int,
                                 help="Choose size of learned representation", default=256)
        self.parser.add_argument("--init_lr", type=float, help="Choose initial learning rate", default=1.0e-4)
        self.parser.add_argument("--decay", type=float, help="Choose decay rate", default=0.95)
        self.parser.add_argument("--dropout", type=float, help="Choose dropout rate", default=0.0)
        self.parser.add_argument("--clip", type=float, help="Choose gradient clipping value", default=0.1)
        self.parser.add_argument("--cutoff", type=float, help="Choose cutoff distance", default=None)
        self.parser.add_argument("--ratio", type=float, help="Choose train:val:test ratio", default=None, nargs="+")

        self.parser.add_argument("--ensemble", action='store_true', help="Train ensemble")
        self.parser.add_argument("--ff", action='store_true', help="Generate 3D geometries via MMFF94")
        self.parser.add_argument("--rdf", action='store_true', help="Use RDF in representation")
        self.parser.add_argument("--cdf", action='store_true', help="Use CDF in representation")
        self.parser.add_argument("--mfd", action='store_true', help="Use MFD in representation")
        self.parser.add_argument("--no_bias", action='store_false', help="Do not use bias")
        self.parser.add_argument("--simple", action='store_true', help="Use simple features in representation")
        self.parser.add_argument("--charge", action='store_true', help="Use charge in representation")
        self.parser.add_argument("--no_hydrogens", action='store_true', help="Remove hydrogens in 3D graph")
        self.parser.add_argument("--mean_readout", action='store_true', help="Use mean readout function else sum")
        self.parser.add_argument("--topology", action='store_false', help="Only use 2D information")
        self.parser.add_argument("--no_scaler", action='store_false', help="Only use 2D information")
        self.parser.add_argument("--save_dl", action='store_true', help="Save DataLoader for later in .pkl file")
        self.parser.add_argument("--test", action='store_true', help="Test external dataset")
        self.parser.add_argument("--transfer", action='store_true', help="Perform transfer learning")
        self.parser.add_argument("--locked_transfer", action='store_true', help="Perform locked transfer learning")
        self.parser.add_argument("--store_models", action='store_true', help="Store trained ANNs")


class InputArguments:
    def __init__(self, training_type="train"):
        # Data processing
        args = ArgParser().parser.parse_args()
        self.save_dl = args.save_dl
        self.data_type = "molecules"  # molecules, reactions, mixtures
        self.training_type = training_type  # training or test
        self.transfer = args.transfer
        self.test = args.test
        self.locked_transfer = True
        self.store_models = args.store_models
        self.dir = path.abspath(path.join("__file__", "../../.."))

        if training_type == "train":
            self.input_file = args.data
        else:
            self.input_file = None

        if args.transfer_data is not None:
            self.transfer_file = args.transfer_data
        else:
            self.transfer_file = None

        if args.test_data is not None:
            self.test_file = args.test_data
        else:
            self.test_file = None

        if args.save_dir is None:
            self.save_dir = self.dir + "/Results/"
        else:
            self.save_dir = args.save_dir

        if "," in args.property:
            self.property = args.property.split(",")
        else:
            self.property = [args.property]

        if args.transfer_property is None:
            self.transfer_property = self.property
        elif "," in args.transfer_property:
            self.transfer_property = args.transfer_property.split(",")
        else:
            self.transfer_property = [args.transfer_property]

        self.ff_3d = args.ff
        self.processors = 10
        self.seed = args.seed
        self.scaler = args.no_scaler

        # Features
        self.include_3d = args.topology
        self.charge = args.charge
        self.rdf = args.rdf
        self.cdf = args.cdf
        self.mfd = args.mfd
        self.simple_features = args.simple
        self.no_hydrogens = args.no_hydrogens
        self.mean_readout = args.mean_readout

        # Hyperparameters

        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.activation = "LeakyReLU"
        self.dropout = args.dropout
        self.bias = args.no_bias
        self.max_epochs = args.epochs  # 1000
        self.patience = args.patience
        self.transfer_patience = args.transfer_patience
        self.batch_size = args.batch
        self.transfer_batch = args.transfer_batch
        if args.ratio is None:
            self.ratio = (0.8, 0.1, 0.1)
        elif sum(args.ratio) != 1:
            raise ValueError(f"Training:validation:test ratio does not sum to 1!")
        else:
            self.ratio = tuple(args.ratio)
        self.init_lr = args.init_lr
        self.clipvalue = args.clip
        self.decay_rate = args.decay
        self.cutoff = args.cutoff
        self.hidden_message = args.hidden_message  # 512
        self.depth = args.depth
        self.representation_size = args.representation_size  # 256

        self.ensemble = args.ensemble
        self.outer_folds = args.folds
        self.inner_folds = 9

        # Plotting

        self.color_1 = "#0F4C81"  # Pantone Classic Blue
        self.color_2 = "#EA733D"  # Pantone Red 032 C
        self.color_3 = "#D01C1F"  # Pantone Fiery Red
        self.font = "Arial"
        self.font_size = 24
