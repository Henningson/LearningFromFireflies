import argparse

# Helper function, such that we can easily share the same argument parser throughout different files.


class GlobalArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog="Learning from Fireflies",
        description="See its name",
        epilog="Stuff.",
    ):
        argparse.ArgumentParser.__init__(self, prog, description, epilog)

        self.add_argument("--config", type=str, default="config.yml")

        self.add_argument("--hle_path", type=str, default="HLE_dataset")
        self.add_argument("--ff_path", type=str, default="fireflies_dataset_v3")

        self.add_argument("--learning_rate", type=float, default=0.1)
        self.add_argument("--batch_size", type=int, default=8)
        self.add_argument("--num_epochs", type=int, default=100)
        self.add_argument("--loss", type=str, default="ce")
        self.add_argument("--eval_keys", type=str, required=False, default="")
        self.add_argument("--train_keys", type=str, required=False, default="")
