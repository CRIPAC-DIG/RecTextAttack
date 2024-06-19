# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import List, Union

# import lru
import torch

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.goal_functions import GoalFunction
from textattack.models.wrappers import ModelWrapper
from textattack.shared import AttackedText, utils


from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import LevenshteinEditDistance, MaxWordsPerturbed
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.search_methods import SearchMethod, GreedyWordSwapWIR


from textattack.transformations import (
    Transformation,
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapMaskedLM,
)



from prompt_attack.attacker import AdvPromptAttack
from prompt_attack.goal_function import PromptGoalFunction, target_item_exposure, encode_one_item
from prompt_attack.label_constraint import LabelConstraint
from prompt_attack.attacker import create_attack
import logging





import os
import json
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset

import warnings

warnings.simplefilter("ignore")


def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval(model, dataloader, args, attack_items=None, return_user_embed=False):

    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()
    user_embeds = []

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch, return_users=return_user_embed)

        if return_user_embed:
            user_embeds.append(scores[1].detach().cpu())
        scores = scores[0]
        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["attack_score"] = scores.mean(dim=0, keepdims=True)[:, attack_items]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()
    average_metrics['attack_score'] = average_metrics['attack_score'].mean()
    if return_user_embed:
        user_embeddings = torch.cat(user_embeds, dim=0)
        return average_metrics, user_embeddings 
    else:
        return average_metrics

def eval_with_user_embedding(model, dataloader, args, attack_items=None, user_embeddings=None):
    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()
    start_idx = 0

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

        labels = labels.to(args.device)
        users = user_embeddings[start_idx:start_idx+len(labels)].to(args.device)
        start_idx += len(labels)


        with torch.no_grad():
            scores = model.similarity_score(pooler_output=users)
        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["attack_score"] = scores.mean(dim=0, keepdims=True)[:, attack_items]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()
    average_metrics['attack_score'] = average_metrics['attack_score'].mean()
    return average_metrics


def main():
    parser = ArgumentParser()
    # path and file
    # parser.add_argument('--pretrain_ckpt', type=str, default='checkpoints/toys/best_model.bin')
    # parser.add_argument('--data_path', type=str, default='finetune_data/toys')
    parser.add_argument('--dataset', type=str, default='beauty')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='./longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
    parser.add_argument('--seed', type=int, default=42)


    # train
    parser.add_argument('--num_train_epochs', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=-1)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)

    # attack
    parser.add_argument('--attack', type=str, default='textfooler', choices=['textfooler', 'textbugger', 'deepwordbug', 'bertattack', 'gpt', 'trival', 'punc'])
    parser.add_argument('--ft', action='store_true')
    

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    args.data_path = 'finetune_data/%s' % args.dataset
    dataset = args.dataset
    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    suffix = ''
    suffix = suffix + '.ft' if args.ft else suffix
    save_path = 'results/%s.%s%s.json' % (args.attack, dataset, suffix)
    print('Load attacked items from:', save_path)
    saved_json = read_result(save_path)

    attacked_items = [int(k) for k in saved_json.keys()]
    print('Attacked items:', attacked_items[:10])

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 1
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')
    if not args.ft:
        args.pretrain_ckpt = 'pretrain_ckpt/recformer_seqrec_ckpt.bin'
        dir_preprocess = path_corpus / ('preprocess_%d' % config.max_attr_num)
    else:
        args.pretrain_ckpt = 'checkpoints/{}/best_model.bin'.format(dataset)
        dir_preprocess = path_corpus / ('preprocess_%d_ft' % config.max_attr_num)
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)
    model = RecformerForSeqRec(config)
    print('Loading pretrain model from {}'.format(args.pretrain_ckpt))
    pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    
    item_embeddings = torch.load(path_item_embeddings, map_location='cpu').to(args.device)

    model.init_item_embedding(item_embeddings)


    test_metrics, test_user_embed = eval(model, test_loader, args, attacked_items, return_user_embed=True)
    print('Ori Test:', test_metrics)


    attacked_item_meta_dict = {}
    for item_id, item_info in saved_json.items():
        if args.attack == 'gpt':
            item_text = item_info['gpt_prompt'][1:-1]
        elif args.attack == 'trival':
            item_text = item_info['trival_prompt'].split('title ')[1]
        else:
            item_text = item_info['attacked_prompt'].split('title ')[1]
        attacked_item_meta_dict[int(item_id)] = {'title': item_text}

    for item_id, item_attr in attacked_item_meta_dict.items():
        input_ids, token_type_ids = tokenizer.encode_item(item_attr)
        tokenized_items[item_id] = [input_ids, token_type_ids]
    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)


    attacked_test_metrics = eval_with_user_embedding(model, test_loader, args, attacked_items, test_user_embed)
    print('Attacked Test:', attacked_test_metrics)
    attacked_test_metrics['attack_score'] = attacked_test_metrics['attack_score'].item()
    # save test metrices into json
    save_path = f'results/results{suffix}.json'
    print(f'Save results to {save_path}')
    new_json = json.load(open(save_path))
    if dataset in new_json:
        new_json[dataset].update({args.attack: attacked_test_metrics})
    else:
        new_json[dataset] = {args.attack: attacked_test_metrics}
    json.dump(new_json, open(save_path, 'w'), indent=4)



def item_exposure(user_embeddings, item_embeddings, item_id, item_text, model):
    item_text = {'title': item_text}
    attacked_embeddings = item_embeddings.clone()
    attacked_embeddings[item_id] = encode_one_item(model.longformer, tokenizer_glb, item_text).to(attacked_embeddings.device)
    ranking_score = target_item_exposure(user_embeddings, attacked_embeddings, [item_id])
    return ranking_score[0]

def read_result(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            result = json.load(f)
    else:
        result = {}
    return result
               
if __name__ == "__main__":
    main()




