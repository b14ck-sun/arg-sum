import pandas as pd
from collections import defaultdict
from os import walk
from random import sample
import random

def get_args_kp_by_topic_argkp():

    test_args = pd.read_csv('./KPA_2021_shared_task/test_data/arguments_test.csv')
    test_kps = pd.read_csv('./KPA_2021_shared_task/test_data/key_points_test.csv')
    test_lbl = pd.read_csv('./KPA_2021_shared_task/test_data/labels_test.csv')
    test_topic_id = test_kps['topic'].unique()
    test_kp_id = test_lbl['key_point_id'].unique()

    args_for_kp_by_topic = defaultdict(dict)

    for topic in test_topic_id:
        topic_rows = test_kps.loc[test_kps['topic'] == topic]
        test_kp_id = topic_rows['key_point_id'].unique()
        args_for_kp_by_topic[topic] = {}
        args_for_kp = defaultdict(dict)
        for id in test_kp_id:
            all_args = test_lbl.loc[test_lbl['key_point_id'] == id]
            args = all_args.loc[all_args['label'] == 1]
            args_for_kp[id] = list(args['arg_id'])
        args_for_kp_by_topic[topic] = args_for_kp
    

    args_kp_by_topic = defaultdict(dict)

    for topic in args_for_kp_by_topic.keys():
        args_kp = defaultdict(list)
        for k in args_for_kp_by_topic[topic].keys():
            arg_ids = args_for_kp_by_topic[topic][k]
            args = []
            for arg_id in arg_ids:
                args.append(test_args.loc[test_args['arg_id'] == arg_id]['argument'].values[0])
            args_kp[test_kps.loc[test_kps['key_point_id'] == k]['key_point'].values[0]] = args
        args_kp_by_topic[topic] = args_kp
    return args_kp_by_topic

def add_other_arg_kp(args_kp_by_topic):
    test_args = pd.read_csv('./KPA_2021_shared_task/test_data/arguments_test.csv')
    for topic in args_kp_by_topic:
        all_args = list(test_args.loc[(test_args['topic'] == topic)]['argument'])
        kp_args = list(args_kp_by_topic[topic].values())
        flat_kp_args = [item for sublist in kp_args for item in sublist]
        for arg in all_args:
            if not arg in flat_kp_args:
                args_kp_by_topic[topic]['Other'].append(arg)
    return args_kp_by_topic

def get_args_kp_by_topic_debate():
    path = './reason/reason/'
    folders = ['abortion', 'gayRights', 'marijuana', 'obama']

    arguments_by_topic = defaultdict(dict)

    for folder in folders:
        filenames = next(walk(path+folder), (None, None, []))[2]
        arguments_by_topic[folder] = {}
        args_by_label = defaultdict(list)
        for filename in filenames:
            with open(path+folder+'/'+filename, encoding='utf-8', errors='ignore') as f:
                label = None
                for line in f:
                    if 'Label##' in line:
                        label = line.removeprefix('Label##').strip()
                    if 'Line##' in line:
                        args_by_label[label].append(line.removeprefix('Line##').strip())
        arguments_by_topic[folder] = args_by_label


    path = './reason/reason/labels/'
    for folder in folders:
        labels = []
        with open(path+folder+'.txt', encoding='utf-8', errors='ignore') as f:
            file_list = []
            for line in f:
                if line.strip(): file_list.append(line.strip())
                # print()
            for i in range(int(len(file_list)/2)):
            # print(file_list[i*2])
                if file_list[i*2] == 'p-other':
                    arguments_by_topic[folder]['Pro Other'] = arguments_by_topic[folder].pop('p-other')
                elif file_list[i*2] == 'p-Other':
                    arguments_by_topic[folder]['Pro Other'] = arguments_by_topic[folder].pop('p-Other')
                elif file_list[i*2] == 'c-other':
                    arguments_by_topic[folder]['Con Other'] = arguments_by_topic[folder].pop('c-other')
                elif file_list[i*2] == 'c-Other':
                    arguments_by_topic[folder]['Con Other'] = arguments_by_topic[folder].pop('c-Other')
                else:
                    arguments_by_topic[folder][file_list[i*2+1]] = arguments_by_topic[folder].pop(file_list[i*2])

    # Remove Other
    for folder in folders:
        arguments_by_topic[folder].pop('Pro Other')
        arguments_by_topic[folder].pop('Con Other')
    
    return arguments_by_topic


def get_limit_ds(args_kp_by_topic, limit):
  args_for_kp_by_topic_limit = defaultdict(dict)
  for topic in args_kp_by_topic:
    for kp in args_kp_by_topic[topic]:
      args_for_kp_by_topic_limit[topic][kp] = args_kp_by_topic[topic][kp][:limit]
  return args_for_kp_by_topic_limit

def limit_output(sum, limits):
    return [sum[i][:limits[i]] for i in range(len(sum))]


def get_limit_percentage_sampling(args_kp_by_topic, limit, seed = 0):
  random.seed(seed)
  if limit>1 or limit<0:
    print("limit has to be between 0 and 1")
  args_for_kp_by_topic_limit = defaultdict(dict)
  for topic in args_kp_by_topic:
    for kp in args_kp_by_topic[topic]:
      lim_num = int(len(args_kp_by_topic[topic][kp])*limit)
      args_for_kp_by_topic_limit[topic][kp] = sample(args_kp_by_topic[topic][kp], lim_num)
  return args_for_kp_by_topic_limit