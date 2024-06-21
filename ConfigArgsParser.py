from printer import Printer


class ConfigArgsParser(dict):
    def __init__(self, config, argparser, *arg, **kw):
        super(ConfigArgsParser, self).__init__(*arg, **kw)
        # We assume config to be a dict
        # First copy it
        for key, value in config.items():
            self[key] = value

        # Next, match every key and value in argparser and overwrite it, if it exists
        for key, value in vars(argparser).items():
            if value is None:
                continue

            if key in config:
                self[key] = value
            else:
                Printer.Warning("Key {0} does not exist in config.".format(key))

    def printFormatted(self):
        for key, value in self.items():
            Printer.KV(key, value)

    def printDifferences(self, config):
        for key, value in self.items():
            if config[key] != value:
                Printer.KV(key, value)
            else:
                Printer.KV2(key, value)
