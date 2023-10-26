import spacy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer
import os
from debater_python_api.api.clients.narrative_generation_client import Polarity
from debater_python_api.api.debater_api import DebaterApi
from debater_python_api.api.sentence_level_index.client.sentence_query_base import SimpleQuery
from debater_python_api.api.sentence_level_index.client.sentence_query_request import SentenceQueryRequest
from sentence_transformers import SentenceTransformer, util
from scipy.special import softmax
from collections import defaultdict
from statistics import mean

def get_true_labels(arguments, args_kp_by_topic, topic):
  keys = list(args_kp_by_topic[topic].keys())
  keys.append('Other')
  label2id = {}
  for i in range(len(keys)):
    label2id[keys[i]] = i

  # print(label2id)
  true_labels = []
  for arg in arguments:
    f = 0
    for k in keys:
      if arg in args_kp_by_topic[topic][k]:
        true_labels.append(label2id[k])
        f = 1
        break
    if f == 0:
      true_labels.append(label2id['Other'])
  return np.array(true_labels)

def cluster_argkp(input_args_kp_by_topic, embedder, limits = []):
  test_args = pd.read_csv('./KPA_2021_shared_task/test_data/arguments_test.csv')
  cluster_by_topic = {}
  for topic in input_args_kp_by_topic:

    arguments = test_args.loc[(test_args['topic'] == topic)]
    arguments = list(arguments['argument'])

    corpus_embeddings = embedder.encode(arguments)
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    if limits:
        n_clusters = limits[list(input_args_kp_by_topic).index(topic)]
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
       clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(arguments[sentence_id])

    cluster_by_topic[topic] = clustered_sentences
    # true_labels = get_true_labels(arguments, args_kp_by_topic, topic)
    # print(adjusted_rand_score(true_labels, cluster_assignment))
  # return cluster_by_topic, true_labels
  return cluster_by_topic, ''

def cluster(input_args_kp_by_topic, embedder, limits = []):
  cluster_by_topic = {}
  for topic in input_args_kp_by_topic:

    arguments = list(set().union(*input_args_kp_by_topic[topic].values()))

    corpus_embeddings = embedder.encode(arguments)
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    if limits:
        n_clusters = limits[list(input_args_kp_by_topic).index(topic)]
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
       clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(arguments[sentence_id])

    cluster_by_topic[topic] = clustered_sentences
    # true_labels = get_true_labels(arguments, args_kp_by_topic, topic)
    # print(adjusted_rand_score(true_labels, cluster_assignment))
  # return cluster_by_topic, true_labels
  return cluster_by_topic, ''

def get_tokenizer():
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
   return tokenizer

def ds(data):
  tokenizer = get_tokenizer()
  MAX_LEN = 512
  token_ids = []
  mask_ids = []
  seg_ids = []
  for prem, hyp in data:
    premise_id = tokenizer.encode(prem, add_special_tokens = False)
    hypothesis_id = tokenizer.encode(hyp, add_special_tokens = False)
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
    premise_len = len(premise_id)
    hypothesis_len = len(hypothesis_id)

    segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))
    attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))
    token_ids.append(torch.tensor(pair_token_ids))
    seg_ids.append(segment_ids)
    mask_ids.append(attention_mask_ids)

  token_ids = pad_sequence(token_ids, batch_first=True)
  mask_ids = pad_sequence(mask_ids, batch_first=True)
  seg_ids = pad_sequence(seg_ids, batch_first=True)
  test_ds = TensorDataset(token_ids, mask_ids, seg_ids)
  return test_ds

def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
    return optimizer

def get_preds(test_loader, model):
    optimizer = get_optimizer(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction = []
    with torch.no_grad():
        for batch_idx, (pair_token_ids, mask_ids, seg_ids) in enumerate(test_loader):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)

            preds = model(pair_token_ids, mask_ids, seg_ids)
            prediction.extend(preds[0].tolist())
    return prediction





def arguments_coverage_short_candidates(arguments_list, model, max_sent_len=70, min_sent_len=3):
  arg_pairs = []
  candidates = [candidate for candidate in arguments_list if len(candidate)<max_sent_len and len(candidate.split())>=min_sent_len]
  if not candidates:
    candidates = arguments_list
  for candidate in candidates:
    for arg in arguments_list:
      arg_pairs.append([candidate, arg])

  dsa = ds(arg_pairs)
  loader = DataLoader(dsa, shuffle=False, batch_size=128)
  preds = get_preds(loader, model)
  soft_preds = softmax(preds, axis=1)
  labeled_preds = []
  for i in range(len(soft_preds)):
    if preds[i][1] > preds[i][0] and preds[i][1] > 0:
      labeled_preds.append(1)
    else:
      labeled_preds.append(0)

  arg_coverage = defaultdict(dict)
  for i in range(len(candidates)):
    s = 0
    s1 = labeled_preds[i*len(arguments_list):(i+1)*len(arguments_list)]
    for ss in s1:
      s += ss
    arg_coverage[candidates[i]] = s
  return arg_coverage

def arguments_coverage_all_candidates(arguments_list, model):
  arg_pairs = [[a, b] for idx, a in enumerate(arguments_list) for b in arguments_list]
  dsa = ds(arg_pairs)
  loader = DataLoader(dsa, shuffle=False, batch_size=128)
  preds = get_preds(loader, model)
  soft_preds = softmax(preds, axis=1)
  labeled_preds = []
  for i in range(len(soft_preds)):
    if preds[i][1] > preds[i][0] and preds[i][1] > 0:
      labeled_preds.append(1)
    else:
      labeled_preds.append(0)

  arg_coverage = defaultdict(dict)
  for i in range(len(arguments_list)):
    s = 0
    s1 = labeled_preds[i*len(arguments_list):(i+1)*len(arguments_list)]
    for ss in s1:
      s += ss
    arg_coverage[arguments_list[i]] = s
  return arg_coverage



def top_arg_scoring_v1(topic, arg_cov):
  scored_candids = defaultdict()
  highes_score = -1
  for arg in arg_cov:
    if not arg or len(arg.split()) == 0:
      continue
    # score = (arg_cov[arg] * 2) / len(arg)
    score = (arg_cov[arg] ** 5) / len(arg.split())
    # score = (math.exp(arg_cov[arg])) / len(arg.split())
    scored_candids[arg] = score
    highes_score = max(highes_score, score)
  candidates = [i for i in scored_candids if scored_candids[i] == highes_score]
  return sorted(candidates, key=len)[0]

def top_arg_shortest(topic, arg_cov):
  highest = max(arg_cov.values())
  candidates = [i for i in arg_cov if arg_cov[i] == highest]
  return sorted(candidates, key=len)[0]

def check_cosine_sim(embedder, arg, candid, thresh=0.9):
  if util.pytorch_cos_sim(embedder.encode(arg), embedder.encode(candid))<thresh:
    return True
  return False


def top_arg_highest_qual_cossim(topic, arg_cov, selected_cadidates, embedder):
  highest = max(arg_cov.values())
  candidates = [i for i in arg_cov if arg_cov[i] == highest]
  scored = score_candidates(topic, candidates)
  sorted_scored = sorted(scored, key=lambda x: x[1], reverse=True)
  for arg in sorted_scored:
    for candid in selected_cadidates:
      if check_cosine_sim(embedder, arg[0], candid):
        return arg[0]
  return sorted_scored[0][0]

def score_candidates(topic, sentences):
    debater_api = DebaterApi('7461b4d34a084f0af9ca7ee25ef2edf2L05') #api_key should be obtained from ibm project debater
    arg_quality_client = debater_api.get_argument_quality_client()
    sentence_topic_dicts = [{'sentence' : sentence, 'topic' : topic } for sentence in sentences]
    scores = arg_quality_client.run(sentence_topic_dicts)
    return list(zip(sentences, scores))

def sort_sums(cluster_summary_by_topic):
  sorted_summaries = []
  for t in cluster_summary_by_topic:
    s = list(cluster_summary_by_topic[t])
    s.sort(reverse=True, key=lambda x: x[1])
    ss = [item[0] for item in s]
    sorted_summaries.append(ss)
  return sorted_summaries


def method_v10(cluster_by_topic, model, embedder):
  nlp = spacy.load("en_core_web_sm")

  cluster_summary_by_topic = {}
  for topic in cluster_by_topic:
    summaries = []
    selected_cadidates = []
    for cluster in cluster_by_topic[topic]:
      arguments = cluster_by_topic[topic][cluster]
      arguments = [str(word) for line in arguments for word in nlp(line).sents]
      arg_and_cov = arguments_coverage_short_candidates(arguments, model)
      sum = top_arg_highest_qual_cossim(topic, arg_and_cov, selected_cadidates, embedder)
      selected_cadidates.append(sum)
      summaries.append([sum, len(arguments)])
    cluster_summary_by_topic[topic] = summaries
  sorted_summaries = sort_sums(cluster_summary_by_topic)
  return sorted_summaries

def method_v8(cluster_by_topic, model, embedder = ''):
  nlp = spacy.load("en_core_web_sm")
  cluster_summary_by_topic = {}
  for topic in cluster_by_topic:
    summaries = []
    for cluster in cluster_by_topic[topic]:
      arguments = cluster_by_topic[topic][cluster]
      arguments = [str(word) for line in arguments for word in nlp(line).sents]
      arg_and_cov = arguments_coverage_short_candidates(arguments, model)
      sum = top_arg_shortest(topic, arg_and_cov)
      summaries.append([sum, len(arguments)])
    cluster_summary_by_topic[topic] = summaries
  sorted_summaries = sort_sums(cluster_summary_by_topic)
  return sorted_summaries

def method_v6(cluster_by_topic, model, embedder = ''):
  cluster_summary_by_topic = {}
  for topic in cluster_by_topic:
    summaries = []
    for cluster in cluster_by_topic[topic]:
      arguments = cluster_by_topic[topic][cluster]
      arguments = [sent for arguments in arguments for sent in arguments.split('.') ]
      arg_and_cov = arguments_coverage_all_candidates(arguments, model)
      sum = top_arg_shortest(topic, arg_and_cov)
      summaries.append([sum, len(arguments)])

    cluster_summary_by_topic[topic] = summaries
  sorted_summaries = sort_sums(cluster_summary_by_topic)
  return sorted_summaries


def method_v11(cluster_by_topic, model, embedder = ''):
  nlp = spacy.load("en_core_web_sm")
  cluster_summary_by_topic = {}
  for topic in cluster_by_topic:
    summaries = []
    for cluster in cluster_by_topic[topic]:
      arguments = cluster_by_topic[topic][cluster]
      arguments = [str(word) for line in arguments for word in nlp(line).sents]
      arguments = [x for x in arguments if x]
      arg_and_cov = arguments_coverage_all_candidates(arguments, model)
      sum = top_arg_scoring_v1(topic, arg_and_cov)
      summaries.append([sum, len(arguments)])

    cluster_summary_by_topic[topic] = summaries
  sorted_summaries = sort_sums(cluster_summary_by_topic)
  return sorted_summaries