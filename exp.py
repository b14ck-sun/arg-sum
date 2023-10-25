from datasets import *
from eval import *
from methods import *
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, AdamW
import torch
import copy
import random
import numpy as np

def run_experiment_argkp(model, embedder, dataset, method, limits = []):
    clusters, true_labels = cluster(dataset, embedder, limits)
    sums = method(clusters, model, embedder)
    limted_outputs = limit_output(sums, limits_argkp)
    return limted_outputs, get_actual_coverage_2(limted_outputs, argkp)

def run_experiment_debate(model, embedder, dataset, method, limits = []):
    clusters, true_labels = cluster(dataset, embedder, limits)
    sums = method(clusters, model, embedder)
    # limted_outputs = limit_output(sums, limits_debate)
    return sums, get_actual_coverage_2(sums, debate)

def print_total_arg_num(ds):
    c = 0
    for topic in ds:
        for kp in ds[topic]:
            c += len(list(ds[topic][kp]))
    print("Total number of arguments is: " + str(c))

def exp_sampled_coverage_argkp(method, seed_id, thresh_limits = [0.25, 0.5, 0.75, 1]):
    method_sums = []
    method_coverages = []
    for l in thresh_limits:
        limited_ds = get_limit_percentage_sampling(argkp_wother, l, seed_id)
        sums, coverage = run_experiment_argkp(model, embedder, limited_ds, method, limits = [])
        method_sums.append(sums)
        method_coverages.append(coverage)
    return method_coverages, method_sums

def exp_sampled_coverage_debate(method, seed_id, thresh_limits = [0.25, 0.5, 0.75, 1]):
    method_sums = []
    method_coverages = []
    for l in thresh_limits:
        limited_ds = get_limit_percentage_sampling(debate, l, seed_id)
        sums, coverage = run_experiment_debate(model, embedder, limited_ds, method, limits = limits_debate)
        method_sums.append(sums)
        method_coverages.append(coverage)
    return method_coverages, method_sums


embedder = SentenceTransformer('./V1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("./2", num_labels=2)
model.to(device)

limits_argkp = [9, 10, 14]
limits_debate = [13, 9, 10, 16]
argkp = get_args_kp_by_topic_argkp()
debate = get_args_kp_by_topic_debate()
argkp_wother = copy.deepcopy(argkp)
argkp_wother = add_other_arg_kp(argkp_wother)


###### Experimenting on ArgKP, method 6 and 11
seed_id = 0

method6_sums = []
method6_coverages = []
method10_sums = []
method10_coverages = []

sums, coverage = run_experiment_argkp(model, embedder, argkp_wother, method_v6, limits = [])
method6_sums.append(sums)
method6_coverages.append(coverage)

sums, coverage = run_experiment_argkp(model, embedder, argkp_wother, method_v11, limits = [])
method10_sums.append(sums)
method10_coverages.append(coverage)




###### Experimentingon debate, method 6 and 11
# Method 6
sums, coverage = run_experiment_debate(model, embedder, debate, method_v6, limits = limits_debate)
print(sums)
print(coverage)

# Method 11
sums, coverage = run_experiment_debate(model, embedder, debate, method_v11, limits = limits_debate)
print(sums)
print(coverage)