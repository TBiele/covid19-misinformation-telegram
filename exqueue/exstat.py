
import os
import subprocess
import argparse
import time
import heapq
from datetime import datetime
import json

from filelock import FileLock


time_format = '%Y%m%d%H%M%S'


def ex_format(ex):
	c_status = ex['current_status']
	p_id = ex['process_id']
	status = c_status['status']
	timestamp = datetime.strptime(c_status['timestamp'], time_format)
	experiment = ex['experiment']
	ex_id = ex['ex_id']
	return f'[{status}] {experiment} {timestamp} ({ex_id}) - {p_id}'


def get_experiments(queue_path, status):
	ex_list = []
	status_path = os.path.join(queue_path, status)
	if not os.path.exists(status_path):
		os.mkdir(status_path)
	for file in os.listdir(status_path):
		file_path = os.path.join(status_path, file)
		if os.path.isfile(file_path) and not file_path.endswith('.lock'):
			with open(file_path, 'r') as f:
				ex = json.load(f)
			c_status = ex['current_status']
			timestamp = datetime.strptime(c_status['timestamp'], time_format)
			ex_list.append((timestamp, ex))
	return ex_list


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-qp', '--queue_path', default='~/.default_queue')
	args = parser.parse_args()

	queue_path = os.path.expanduser(args.queue_path)

	if not os.path.exists(queue_path):
		os.mkdir(queue_path)

	with FileLock(os.path.join(queue_path, '.lock')):
		ex_queue = {
			'submitted': get_experiments(queue_path, 'submitted'),
			'running': get_experiments(queue_path, 'running'),
			'completed': get_experiments(queue_path, 'running')
		}

		for ts, ex in sorted(ex_queue['running'], key=lambda x: x[0], reverse=True):
			print(f'{ex_format(ex)}')
		print(f'--------------------------')
		for ts, ex in sorted(ex_queue['submitted'], key=lambda x: x[0], reverse=True):
			print(f'{ex_format(ex)}')
		print(f'--------------------------')
		for ts, ex in sorted(ex_queue['completed'], key=lambda x: x[0], reverse=True):
			print(f'\t{ex_format(ex)}')



