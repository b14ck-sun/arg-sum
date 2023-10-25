import pandas as pd
from collections import defaultdict
from statistics import mean

def get_actual_duplicates(sums, args_kp_by_topic):
  topics = list(args_kp_by_topic.keys())
  duplicates = []
  for topic_id in range(len(sums)):
    covered_kps = []
    kps = list(args_kp_by_topic[topics[topic_id]].keys())
    for arg in sums[topic_id]:
      found = 0
      for kp in kps:
         if arg in args_kp_by_topic[topics[topic_id]][kp]:
          covered_kps.append(kp)
          found = 1
      if found == 0:
        covered_kps.append('None')
    duplicates.append(len(covered_kps) - len(set(covered_kps)))
  print(sum(duplicates))
  return duplicates

def get_actual_duplicates_all_unique(sums, args_kp_by_topic):
  topics = list(args_kp_by_topic.keys())
  duplicates = []
  for topic_id in range(len(sums)):
    covered_kps = []
    kps = list(args_kp_by_topic[topics[topic_id]].keys())
    for arg in sums[topic_id]:
      for kp in kps:
         if arg in args_kp_by_topic[topics[topic_id]][kp]:
          covered_kps.append(kp)
    duplicates.append(len(covered_kps) - len(set(covered_kps)))
  print(sum(duplicates))
  return duplicates

def get_actual_coverage(sums, args_kp_by_topic):
  topics = list(args_kp_by_topic.keys())
  coverages = []
  for topic_id in range(len(sums)):
    covered_kps = set()
    kps = list(args_kp_by_topic[topics[topic_id]].keys())
    for arg in sums[topic_id]:
      for kp in kps:
        if arg in args_kp_by_topic[topics[topic_id]][kp]:
          covered_kps.add(kp)
    coverages.append(len(covered_kps)/len(kps))

  return coverages

def get_actual_coverage_2(sums, args_kp_by_topic):
  topics = list(args_kp_by_topic.keys())
  coverages = []
  for topic_id in range(len(sums)):
    covered_kps = set()
    kps = list(args_kp_by_topic[topics[topic_id]].keys())
    for arg in sums[topic_id]:
      for kp in kps:
        for a in args_kp_by_topic[topics[topic_id]][kp]:
          if arg in a:
            covered_kps.add(kp)
    coverages.append(len(covered_kps)/len(kps))

  return coverages