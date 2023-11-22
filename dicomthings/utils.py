import os
import json


class SortedDict(dict):
    """A dictionary that iters its items/keys/values in sorted key order.

    This dictionary subclass maintains itself in ascending order based on the keys.
    """

    def __init__(self):
        super(SortedDict, self).__init__()

    def __iter__(self):
        return iter(sorted(super(SortedDict, self).__iter__()))

    def items(self):
        return [(k, self[k]) for k in self]

    def keys(self):
        return [k for k in self]

    def values(self):
        return [self[k] for k in self]


class JsonDict(SortedDict):
    """A SortedDict that can be read/written from/to a JSON file.
    """

    def __init__(self):
        super(JsonDict, self).__init__()

    def write(self, file_path, **dump_json_kwargs):
        JsonDict.dump_json(self, file_path, **dump_json_kwargs)

    @staticmethod
    def dump_json(json_dict, file_path, overwrite=False, makedirs=True, ensure_ascii=True):
        assert not os.path.isfile(file_path) or overwrite, "By default an overwrite is not allowed. If wanted specify overwrite=True."
        file_dir = os.path.dirname(file_path)
        assert makedirs or os.path.isdir(file_dir), "The file directory does not exist and you specified makedirs=False."
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        with open(file_path, "w") as f:
            json.dump({k: json_dict[k] for k in json_dict}, f, indent=4, sort_keys=True, ensure_ascii=ensure_ascii)

    @classmethod
    def load_json(cls, file_path):
        assert os.path.isfile(file_path), "The given file path does not exist."
        d = cls()
        with open(file_path, "r") as f:
            d_ = json.load(f)
            for k in list(d_.keys()):
                d[k] = d_.pop(k)
        
        return d


class DotNotationDict(SortedDict):
    """A SortedDict but the attributes can be accessed via dot notation also.
    """
    
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self.__setitem__(attr, value)

    def __delattr__(self, attr):
        self.__delitem__(attr)


class DotJsonDict(JsonDict, DotNotationDict):
    """Combination of a JsonDict and a DotNotationDict.
    """
    
    pass
