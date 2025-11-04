import numpy as np

from patterns.singleton_meta import SingletonMeta


class LabelEncoder(metaclass=SingletonMeta):
    def __init__(self):
        self._classes = None

    def configure_labels(self, unique_labels):
        self._classes = sorted(unique_labels)
        self._label_to_id = {label: i for i, label in enumerate(self._classes)}
        self._id_to_label = {id: label for label, id in self._label_to_id.items()}

        self._classes = np.array(self._classes)

    def _check_labels(self):
        if self._classes is None:
            raise RuntimeError("LabelEncoder not configured")

    def encode(self, labels):
        self._check_labels()

        if isinstance(labels, str):
            return self._label_to_id[labels]
        return np.array([self._label_to_id[label] for label in labels])

    def decode(self, ids):
        self._check_labels()

        if isinstance(ids, int):
            return self._id_to_label[ids]
        return np.array([self._id_to_label[id] for id in ids])

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        assert self._classes is not None
        return len(self._classes)
