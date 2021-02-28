
import os
import json
import argparse
import logging
import torch

from metric_utils import compute_threshold_f1
from data_utils import read_jsonl, label_text_to_relevant_id


def create_dataset(tweets, misinfo, tweet_scores):
	scores = torch.zeros([len(tweets), len(misinfo)], dtype=torch.float)
	labels = torch.zeros([len(tweets), len(misinfo)], dtype=torch.long)
	m_map = {m_id: m_idx for (m_idx, m_id) in enumerate(misinfo.keys())}
	missing_count = 0
	for t_idx, t in enumerate(tweets):
		tweet_id = t['id']
		if tweet_id not in tweet_scores:
			missing_count += 1
			continue
		t_scores = tweet_scores[tweet_id]
		for m_id in misinfo:
			if m_id not in t_scores:
				m_score = 0.0
			else:
				m_score = t_scores[m_id]
			m_label = 0
			if m_id in t['misinfo']:
				m_label = label_text_to_relevant_id(t['misinfo'][m_id])

			labels[t_idx, m_map[m_id]] = m_label
			scores[t_idx, m_map[m_id]] = m_score
	return labels, scores, missing_count


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-tp', '--train_path', required=True)
	parser.add_argument('-vp', '--val_path', required=True)

	parser.add_argument('-tsp', '--train_score_path', default='data/scores.json')
	parser.add_argument('-vsp', '--val_score_path', default='data/scores.json')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-mn', '--model_name', default='covid-twitter-v2-bertscore')
	parser.add_argument('-mip', '--misinfo_path', default=None)
	parser.add_argument('-th', '--threshold', default=None, type=float)
	parser.add_argument('-tm', '--threshold_min', default=0.0, type=float)
	parser.add_argument('-tx', '--threshold_max', default=1.0, type=float)
	parser.add_argument('-ts', '--threshold_step', default=0.0005, type=float)

	args = parser.parse_args()

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	logging.info(f'Loading misinfo: {args.misinfo_path}')
	with open(args.misinfo_path, 'r') as f:
		misinfo = json.load(f)

	logging.info(f'Loading train bertscore scores: {args.train_score_path}')
	with open(args.train_score_path, 'r') as f:
		train_scores = json.load(f)

	logging.info(f'Loading val bertscore scores: {args.val_score_path}')
	with open(args.val_score_path, 'r') as f:
		val_scores = json.load(f)

	logging.info(f'Loading train dataset: {args.train_path}')
	train_data = read_jsonl(args.train_path)
	logging.info(f'Loading val dataset: {args.val_path}')
	val_data = read_jsonl(args.val_path)

	threshold = args.threshold
	if threshold is None:
		logging.info(f'Calculating training threshold...')
		t_labels, t_scores, t_missing = create_dataset(train_data, misinfo, train_scores)
		logging.info(f'Missing training tweet scores: {t_missing}')

		t_f1, t_p, t_r, threshold = compute_threshold_f1(
			scores=t_scores,
			labels=t_labels,
			threshold_min=args.threshold_min,
			threshold_max=args.threshold_max,
			threshold_step=args.threshold_step
		)
		# print(f'{t_p:.4f}\t{t_r:.4f}\t{t_f1:.4f}\t{threshold}')

	logging.info(f'Predicting on val data...')
	v_labels, v_scores, v_missing = create_dataset(val_data, misinfo, val_scores)
	logging.info(f'Missing val tweet scores: {v_missing}')
	f1, p, r, _ = compute_threshold_f1(
		scores=v_scores,
		labels=v_labels,
		threshold=threshold
	)
	print(f'{f1:.4f}\t{p:.4f}\t{r:.4f}\t{threshold}')
