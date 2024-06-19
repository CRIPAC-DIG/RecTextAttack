# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from textattack.constraints import PreTransformationConstraint

class LabelConstraint(PreTransformationConstraint):
    """
    A constraint that does not allow to attack the labels (or any words that is important for tasks) in the prompt.
    """

    def __init__(self, labels=[]):
        self.labels = [label.lower() for label in labels]

    def _get_modifiable_indices(self, current_text):
        modifiable_indices = set()
        for i, word in enumerate(current_text.words):
            if str(word).lower() not in self.labels:
                modifiable_indices.add(i)
        
        return modifiable_indices

    def check_compatibility(self, transformation):
        """
        It is always true.
        """
        return True
    
"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

# hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")

from sentence_transformers import SentenceTransformer

class UniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", device='cpu', **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = SentenceTransformer('stsb-mpnet-base-v2', device=device)
    def encode(self, sentences):
        encoding = self.model.encode(sentences)
        return encoding #.numpy()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None


# import tensorflow as tf
# import tensorflow_hub as hub
# class UniversalSentenceEncoder(SentenceEncoder):
#     """Constraint using similarity between sentence encodings of x and x_adv
#     where the text embeddings are created using the Universal Sentence
#     Encoder."""

#     def __init__(self, threshold=0.8, large=False, metric="angular", device=3, **kwargs):
#         super().__init__(threshold=threshold, metric=metric, **kwargs)
#         if large:
#             tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
#         else:
#             tfhub_url = "/data0/zhangjinghao/tensorflow_hub/universal-sentence-encoder/"

#         self._tfhub_url = tfhub_url
#         # Lazily load the model
#         self.model = None
#         self.device = '/GPU:' + str(device)
#         self.model = hub.load(self._tfhub_url)

#     def encode(self, sentences):
#         with tf.device(self.device):
#             encoding = self.model(sentences)

#             if isinstance(encoding, dict):
#                 encoding = encoding["outputs"]

#             return encoding.numpy()

#     def __getstate__(self):
#         state = self.__dict__.copy()
#         state["model"] = None
#         return state

#     def __setstate__(self, state):
#         self.__dict__ = state
#         self.model = None
