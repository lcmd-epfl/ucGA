import importlib

class Wrapper:
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)
