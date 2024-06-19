# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod

import lru
import numpy as np
import torch

from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.shared.utils import ReprMixin
from textattack.goal_function_results import ClassificationGoalFunctionResult
from sklearn.metrics.pairwise import cosine_similarity


class GoalFunction(ReprMixin, ABC):
    """Evaluates how well a perturbed attacked_text object is achieving a
    specified goal.

    Args:
        model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
            The victim model to attack.
        maximizable(:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the goal function is maximizable, as opposed to a boolean result of success or failure.
        query_budget (:obj:`float`, `optional`, defaults to :obj:`float("in")`):
            The maximum number of model queries allowed.
        model_cache_size (:obj:`int`, `optional`, defaults to :obj:`2**20`):
            The maximum number of items to keep in the model results cache at once.
    """

    def __init__(
        self,
        model_wrapper,
        maximizable=False,
        use_cache=True,
        query_budget=float("inf"),
        model_batch_size=32,
        model_cache_size=2**20,
    ):
        self.model = model_wrapper
        self.maximizable = maximizable
        self.use_cache = use_cache
        self.query_budget = query_budget
        self.batch_size = model_batch_size
        self.ground_truth_output = -1
        if self.use_cache:
            self._call_model_cache = lru.LRU(model_cache_size)
        else:
            self._call_model_cache = None

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def init_attack_example(self, attacked_text):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.num_queries = 0
        # TODO: not skip when init
        result, _ = self.get_result(attacked_text, check_skip=False)
        self.ground_truth_output = result.output
        return result, _

    def get_output(self, attacked_text):
        """Returns output for display based on the result of calling the
        model."""
        return self._get_displayed_output(self._call_model([attacked_text])[0])

    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        # [attacked_text] instead of attacked_text: 
        #   get_results() takes a list of (disturbed) attacked_text
        #   disturbed attacked_text call get_result() directly
        results, search_over = self.get_results([attacked_text], **kwargs) 
        result = results[0] if len(results) else None
        return result, search_over

    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        #TODO: fix bug: len(model_outputs) != len(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def _get_goal_status(self, model_output, attacked_text, check_skip=False):
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    @abstractmethod
    def _is_goal_complete(self, model_output, attacked_text):
        raise NotImplementedError()

    def _should_skip(self, model_output, attacked_text):
        return self._is_goal_complete(model_output, attacked_text)

    @abstractmethod
    def _get_score(self, model_output, attacked_text):
        raise NotImplementedError()

    def _get_displayed_output(self, raw_output):
        return raw_output

    @abstractmethod
    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        raise NotImplementedError()

    @abstractmethod
    def _process_model_outputs(self, inputs, outputs):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to make sure they have a softmax applied.
        """
        raise NotImplementedError()

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        inputs = [at.tokenizer_input for at in attacked_text_list]
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            batch_preds = self.model(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)

    def _call_model(self, attacked_text_list):
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
        if not self.use_cache:
            return self._call_model_uncached(attacked_text_list)
        else:
            uncached_list = []
            for text in attacked_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    uncached_list.append(text)
            uncached_list = [
                text
                for text in attacked_text_list
                if text not in self._call_model_cache
            ]
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs

    def extra_repr_keys(self):
        attrs = []
        if self.query_budget < float("inf"):
            attrs.append("query_budget")
        if self.maximizable:
            attrs.append("maximizable")
        return attrs

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.use_cache:
            state["_call_model_cache"] = self._call_model_cache.get_size()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.use_cache:
            self._call_model_cache = lru.LRU(state["_call_model_cache"])




def encode_one_item(model, tokenizer, item_text):
    model.eval()
    inputs = tokenizer.batch_encode([[item_text]], encode_item=True)
    for k, v in inputs.items():
        inputs[k] = torch.LongTensor(v).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        item_embedding = outputs.pooler_output.detach()
    return item_embedding

class PromptGoalFunction(GoalFunction):

    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def __init__(self, inference, query_budget, verbose=True, logger=None, *args, target_max_acc=0, tokenizer=None, user_embedding=None, item_embedding=None, target_improve=0.05, **kwargs):
        self.inference = inference
        self.query_budget = query_budget
        self.target_max_acc = target_max_acc
        self.logger = logger
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.similarity = cosine_similarity(user_embedding.cpu(), item_embedding.cpu())
        self.attack_cnt = 0
        self.target_improve = target_improve
        super().__init__(*args, **kwargs)
    
    def _clear_cnt(self):
        self.attack_cnt = 0

    def _process_model_outputs(self, inputs, outputs):
        return outputs

    def _is_goal_complete(self, acc, _):
        
        # return acc < self.target_max_acc
        return acc > self.ground_truth_output + self.target_improve
        
    def _get_score(self, model_output, _):
        # Evaluate how much does the adv prompt decrease the accuracy.
        # The higher the better.
        # score = (model_output - self.ground_truth_output).mean()
        score = model_output.mean()
        return score

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        #TODO: is classification?
        return ClassificationGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return raw_output

    def _call_model(self, text_list):
        attr_names = ['title']
        attacked_item_meta_dict = {}
        score_list = []
        for attacked_text in text_list:
            for item_id, item_text in attacked_text._text_input.items():
                item_text = item_text.split(' ')
                attr_idx = [idx for idx, string in enumerate(item_text) for attr in attr_names if attr in string]
                attacked_item_meta_dict[int(item_id)] = {attr_names[i]: ' '.join(item_text[attr_idx[i]+1:attr_idx[i+1]]) if i < len(attr_names)-1 else ' '.join(item_text[attr_idx[i]+1:]) for i in range(len(attr_names))}
            model_outputs = []    
            for item_id, item_text in attacked_item_meta_dict.items():
                if self.verbose:
                    self.logger.info("Current prompt is: {}".format(str(item_text)))
                # query the model
                item_embeddings = self.item_embedding.clone()
                item_embedding = encode_one_item(self.inference.longformer, self.tokenizer, item_text).to(item_embeddings.device)
                item_embeddings[item_id] = item_embedding
                ranking_score = target_item_exposure(self.user_embedding, item_embeddings, [item_id])
                self.attack_cnt += 1

                if self.verbose:
                    self.logger.info("Current ranking score: {:.7f}".format(ranking_score[0]))
                model_outputs += ranking_score
            score_list.append(sum(model_outputs))
        return score_list

    def _get_goal_status(self, model_output, text, check_skip=False):
        return super()._get_goal_status(model_output, text, check_skip)


def create_goal_function(args, inference_model):
    goal_function = PromptGoalFunction(inference=inference_model, 
                                       query_budget=args.query_budget,
                                       logger=args.logger, 
                                       model_wrapper=None, 
                                       verbose=args.verbose)
    return goal_function

def target_item_exposure(user_embeddings, item_embeddings, target_item_ids, k=50):
    similarity_scores = cosine_similarity(user_embeddings.cpu(), item_embeddings.cpu())
    top_items = np.argpartition(-similarity_scores, k)[:, :k]
    exposure_probabilities = [np.mean(np.isin(top_items, item).sum(axis=1)) for item in target_item_ids]
    return exposure_probabilities
