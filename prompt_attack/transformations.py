# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from textattack.transformations import Transformation

class CheckListTransformation(Transformation):

    def generate_random_sequences(num, len):
        seqs = []
        import random
        import string

        for _ in range(num):
            seq = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=len))
            seqs.append(seq)

        return seqs
    
    def _get_transformations(self, current_text, indices_to_modify):
        
        # rand_seqs = self.generate_random_sequences(50, 10)

        rand_seqs = ['d6ZQ3u0GBQ', 'vTAjHynoIG', 'OB4KVJzIft', 'LkF0FZxMZ4', 'iia2yL9Uzm', 'CuVpbbkC4c', 
                     'w52rwgo0Av', 'Vq3aBzuZcD', 'hXLpw3bbiw', 'RcRneWo6Iv', 'S6oUV5E54P', 'xikCjkMydH', 
                     'MQnugHcaoy', 'Q47Jmd4lMV', '9vGXgnbWB8', 'IhuBIhoPGc', '5yWbBXztUY', 'AMsRIKZniY', 
                     'EAB4KP2NVY', '9Q3S1F94fE', 'b74X5IVXQY', 'SFPCqpiClT', 'bhrRSokrfa', 'YHQiDfFhw4', 
                     'BWmsLx8xOA', 'PDCGfz6DL9', 'yh912BU8T0', 'ofOQXLxiW4', 'Cev0s49fbe', 'rzu98kF2KF', 
                     'zexKUTWJYG', '5XeflW1ZJc', 'is77sOXAu8', 'XStsD2pUzu', 'fwhUZUQzXW', 'Pbl3tYuoRo', 
                     'MSAddJ4D2a', 'mzjVogkRhM', 'Kw6nrs57gH', 'ItGDrrA1Sc', 'KjPJJ2a7RB', 'mOJ9nEwT4f', 
                     'ofw9fEkN5R', 'njCuciQJpB', '6a4Yn3RGVc', 'SvAp8RlOFn', 'g0vBZf3tQC', 'zq0DcZ5dnI', 
                     'lf8wBa2yEm', 'lWJoGGRdjv']

        transformed_texts = []
        for rand_seq in rand_seqs:
            transformed_texts.append(current_text.insert_text_after_word_index(index=len(current_text.words)-1, text=rand_seq))
        return transformed_texts        


class StressTestTransformation(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        texts = [" and true is true ", " and false is not true ", " and true is true "*5]
        transformed_texts = []
        for text in texts:
            transformed_texts.append(current_text.insert_text_after_word_index(index=len(current_text.words)-1, text=text))
            # transformed_texts.append(current_text.insert_text_after_word_index(index=0, text=text))

        return transformed_texts 


"""
Word Swap by Random Character Insertion
==========================================================

"""
import numpy as np

import random
import string
import sys

from textattack.transformations import Transformation


class WordSwap(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """

    def __init__(self, letters_to_insert=None):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters

    def _get_special_symbol(self,method='none'):
        # get label of sample
        if method == 'by_sample':
            return self._get_sample_special_symbol()
        else:
            label = self._get_transformation_sample_label()
            if label == 1:
                #negative label = positive symbol
                return '_'

            elif label == 0:
                return '-'
            else:
                print ('error label is not 0 or 1')
                sys.exit('error')

                #positive label = negative symbol

            return None

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        raise NotImplementedError()

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_i_letter(self,letter):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return self.letters_to_insert[letter]
    def _num_letters(self):
        return len(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):

        words = current_text.words
        # print ('current_text',current_text.attack_attrs,list(indices_to_modify))
        transformed_texts = []

        # if 'modified_index' in current_text.attack_attrs:
        #     current_text.attack_attrs['modified_index'].append(list(indices_to_modify)[0])
        # else:
        #     current_text.attack_attrs['modified_index'] = [list(indices_to_modify)[0]]



        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)

            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                modified_text = current_text.replace_word_at_index(i, r)

                if 'modified_index' in modified_text.attack_attrs:
                    modified_text.attack_attrs['modified_index'].append(list(indices_to_modify)[0])
                    if 'modified_index' in current_text.attack_attrs:
                        modified_text.attack_attrs['modified_index'].extend(current_text.attack_attrs['modified_index'])
                else:
                    modified_text.attack_attrs['modified_index'] = [list(indices_to_modify)[0]]
                    if 'modified_index' in current_text.attack_attrs:
                        modified_text.attack_attrs['modified_index'].extend(current_text.attack_attrs['modified_index'])


                transformed_texts_idx.append(modified_text)
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts


class WordSwapTokenSpecificPunctuationInsertion(WordSwap):
    """Transforms an input by inserting a random character.

    random_one (bool): Whether to return a single word with a random
    character deleted. If not, returns all possible options.
    skip_first_char (bool): Whether to disregard inserting as the first
    character. skip_last_char (bool): Whether to disregard inserting as
    the last character.
    """

    def __init__(
        self, all_char=False,random_one=True, skip_first_char=False, skip_last_char=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.random_one = random_one
        self.all_char = all_char
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def _get_replacement_words(self, word): #pass label and special symbols
        """Returns returns a list containing all possible words with 1 random
        character inserted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 1) if self.skip_last_char else len(word)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            if self.all_char == False:
                i = np.random.randint(start_idx, end_idx)
                candidate_word = word[:i] + self._get_random_letter() + word[i:]
                candidate_words.append(candidate_word)
            else:
                i = np.random.randint(start_idx, end_idx)
                for j in range(self._num_letters()):
                    candidate_word = word[:i] + self._get_i_letter(j) + word[i:]
                    candidate_words.append(candidate_word)
        else:
            if self.all_char == False:
                for i in range(start_idx, end_idx):
                    candidate_word = word[:i] + self._get_random_letter() + word[i:]
                    candidate_words.append(candidate_word)
            else:
                for i in range(start_idx, end_idx):
                    for j in range(self._num_letters()):
                        candidate_word = word[:i] + self._get_i_letter(j) + word[i:]
                        candidate_words.append(candidate_word)
        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
