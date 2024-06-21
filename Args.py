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

        self.add_argument("--hle_path", type=str)
        self.add_argument("--ff_path", type=str)

        self.add_argument("--learning_rate", type=float)
        self.add_argument("--batch_size", type=int)
        self.add_argument("--num_epochs", type=int)
