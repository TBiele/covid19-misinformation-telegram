
import argparse
import logging

# pytorch_lightning can cause issues if
# torch or other torch libraries are imported first
# noinspection PyUnresolvedReferences
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from model_utils import *
from data_utils import *

import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-tp', '--train_path', required=True)
	parser.add_argument('-vp', '--val_path', required=True)
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-bs', '--batch_size', default=8, type=int)
	parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=96, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-eo', '--epochs', default=10, type=int)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-lr', '--learning_rate', default=5e-6, type=float)
	parser.add_argument('-lrw', '--lr_warmup', default=0.1, type=float)
	parser.add_argument('-ga', '--gamma', default=0.5, type=float)
	parser.add_argument('-gpu', '--gpus', default='0')
	parser.add_argument('-lt', '--load_checkpoint', default=None)
	parser.add_argument('-tmp', '--train_misinfo_path', default=None)
	parser.add_argument('-vmp', '--val_misinfo_path', default=None)
	parser.add_argument('-es', '--emb_size', default=100, type=int)
	parser.add_argument('-eln', '--emb_loss_norm', default=2, type=int)
	parser.add_argument('-em', '--emb_model', default='transd')
	parser.add_argument('-mt', '--model_type', default='bert')
	parser.add_argument('-mtl', '--model_layers', default=1, type=int)
	parser.add_argument('-wd', '--weight_decay', default=0.0, type=float)
	parser.add_argument('-gcv', '--gradient_clip_val', default=1.0, type=float)
	parser.add_argument('-th', '--threshold', default=None, type=float)

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.makedirs(save_directory)

	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [int(x) for x in args.gpus.split(',')]
	is_distributed = len(gpus) > 1
	precision = 16 if args.use_tpus else 32
	# precision = 32
	tpu_cores = 8
	num_workers = 4
	deterministic = True

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
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

	logging.info(f'Loading tokenizer: {args.pre_model_name}')
	tokenizer = BertTokenizerFast.from_pretrained(args.pre_model_name)
	logging.info(f'Loading train dataset: {args.train_path}')
	train_data = read_jsonl(args.train_path)
	logging.info(f'Loading val dataset: {args.val_path}')
	val_data = read_jsonl(args.val_path)

	logging.info('Loading misinfo')
	with open(args.train_misinfo_path, 'r') as f:
		train_misinfo = json.load(f)
	with open(args.val_misinfo_path, 'r') as f:
		val_misinfo = json.load(f)

		logging.info(f'Loaded misconception info.')
	logging.info('Loading datasets...')

	train_dataset = MisinfoDataset(
		documents=train_data,
		tokenizer=tokenizer,
		misinfo=train_misinfo,
		pos_samples=1,
		neg_samples=1,
		shuffle=True,
	)
	train_data_loader = DataLoader(
		train_dataset,
		num_workers=num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		collate_fn=MisinfoBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		),
		worker_init_fn=train_dataset.worker_init_fn
	)
	train_size = len(train_dataset)

	val_triplet_dataset = MisinfoDataset(
		documents=val_data,
		tokenizer=tokenizer,
		misinfo=val_misinfo,
		pos_samples=1,
		neg_samples=1,
		shuffle=False,
	)
	val_triplet_data_loader = DataLoader(
		val_triplet_dataset,
		num_workers=num_workers,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=MisinfoBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		),
		worker_init_fn=val_triplet_dataset.worker_init_fn
	)

	val_entity_dataset = MisinfoEntityDataset(
		documents=val_data,
		tokenizer=tokenizer,
		misinfo=val_misinfo
	)

	val_rel_dataset = MisinfoRelDataset(
		misinfo=val_misinfo,
		tokenizer=tokenizer,
		m_examples=val_entity_dataset.m_examples
	)

	val_entity_data_loader = DataLoader(
		val_entity_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)
	val_rel_data_loader = DataLoader(
		val_rel_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)

	logging.info(f'train_labels={train_dataset.num_labels}')
	logging.info(f'train={train_size}')
	logging.info(f'val_triplets={len(val_triplet_dataset)}')
	logging.info(f'val_entities={len(val_entity_dataset)}')
	logging.info(f'val_rels={len(val_rel_dataset)}')

	num_batches_per_step = (len(gpus) if not args.use_tpus else tpu_cores)
	updates_epoch = train_size // (args.batch_size * num_batches_per_step)
	updates_total = updates_epoch * args.epochs

	logging.info('Loading model...')
	model = CovidTwitterMisinfoModel(
		pre_model_name=args.pre_model_name,
		learning_rate=args.learning_rate,
		lr_warmup=args.lr_warmup,
		updates_total=updates_total,
		weight_decay=args.weight_decay,
		threshold=args.threshold,
		emb_model=args.emb_model,
		model_type=args.model_type,
		model_layers=args.model_layers,
		emb_size=args.emb_size,
		emb_loss_norm=args.emb_loss_norm,
		gamma=args.gamma,
		load_pretrained=args.load_checkpoint is not None,
	)

	tokenizer.save_pretrained(save_directory)
	model.config.save_pretrained(save_directory)
	if args.load_checkpoint is not None:
		# load checkpoint from pre-trained model
		logging.warning(f'Loading weights from trained checkpoint: {args.load_checkpoint}...')
		model.load_state_dict(torch.load(args.load_checkpoint))

	logger = pl_loggers.TensorBoardLogger(
		save_dir=save_directory,
		flush_secs=30,
		max_queue=2
	)

	if args.use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			logger=logger,
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=args.epochs,
			precision=precision,
			deterministic=deterministic,
			checkpoint_callback=False,
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			logger=logger,
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=args.epochs,
			precision=precision,
			distributed_backend=backend,
			gradient_clip_val=args.gradient_clip_val,
			deterministic=deterministic,
			checkpoint_callback=False,
		)
	try:
		logging.info('Training...')
		trainer.fit(
			model,
			train_data_loader,
			val_dataloaders=[val_triplet_data_loader, val_entity_data_loader, val_rel_data_loader]
		)

		device_id = get_device_id()
		if device_id == 0 or (isinstance(device_id, str) and '0' in device_id):
			logging.info(f'Saving checkpoint on device {device_id}...')
			model.to('cpu')
			torch.save(model.state_dict(), checkpoint_path)

	except Exception:
		logging.exception('Exception while training:')


