"""
Usage:
	python3 test_model.py <DATASET> <MODALITY> <VIDEO_FOR_TEST> <CHECKPOINTS_FILE> \
		--arch <ARCH> --save_scores <SAVE_SCORES>
e.g:
	python3 test_models.py saag01 ARP ./fighting.mkv checkpoints/SAAG01_BNI_ARP_3_model_best.pth.tar \
		--arch BNInception --save_scores scores/ --num_segments 3 --consensus_type avg 
"""

import argparse
import time

# Computation libs
import numpy as np
import cv2
import torch.nn.parallel
import torch.optim
from multiprocessing import Pool, current_process
from sklearn.metrics import confusion_matrix
from PIL import Image

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

from convert_to_ARP import cvApproxRankPooling, Buffer

import torch
import torchsummary


def get_options():
	parser = argparse.ArgumentParser(
		description="Standard video-level testing")
	parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'saag01'])
	parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'ARP'])
	parser.add_argument('video_path', type=str)
	parser.add_argument('checkpoint', type=str)
	parser.add_argument('--arch', type=str, default="resnet101")
	parser.add_argument('--save_scores', type=str, default=None)
	parser.add_argument('--num_segments', type=int, default=3)
	parser.add_argument('--max_num', type=int, default=-1)
	parser.add_argument('--test_crops', type=int, default=10)
	parser.add_argument('--input_size', type=int, default=224)
	parser.add_argument('--consensus_type', type=str, default='avg',
						choices=['avg', 'max', 'topk'])
	parser.add_argument('--k', type=int, default=3)
	parser.add_argument('--dropout', type=float, default=0.7)
	parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
						help='number of data loading workers (default: 5)')
	parser.add_argument('--print_freq', type=int, default=5)

	return parser.parse_args()


def forward_pass_model(model, processed_input):
	return model(processed_input)

def display_prediction():
	pass

def run_video_appx_rank_pooling(
	video_path,
	num_segments
):
	"""Approximated Rank Pooling (ARP) runner for video input

	Outputs Rank pooled frames from a video.
	"""
	current = current_process()
	cap = cv2.VideoCapture(video_path)

	# Find OpenCV version
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		
	# With webcam get(CV_CAP_PROP_FPS) does not work.
	# Let's see for ourselves.	 
	if int(major_ver)  < 3 :
		num_frames = video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
		print ".. Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
	else :
		num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
		print ".. Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

	# Number of frames to capture
    buffer_size = (num_frames+1)//3;

	# Number of frames to capture
	buffer = Buffer(num_frames)
	success = True

	rank_pooled_frames = []
	while success:
		success, frame = cap.read()

		if buffer.isfull():
			frames = buffer.clear()
			rank_pooled = cvApproxRankPooling(frames)
			rank_pooled = Image.fromarray(np.uint8(rank_pooled))
			rank_pooled_frames.append(rank_pooled)

		buffer.enqueue(frame)

	cap.release()
	return rank_pooled_frames

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def sample_frames(frames, sample_indices, new_length):
	sampled_frames = list()
	for idx in sample_indices:
		p = int(idx)
		for _ in range(new_length):
			seg_imgs = frames[p]
			sampled_frames.append(seg_imgs)
			if p < num_frames:
				p += 1
	return sampled_frames
	
def generate_sample_indices(num_frames, new_length, num_segments):
	tick = (num_frames - new_length + 1) / float(num_segments)
	offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])

	return offsets + 1

def transforms_frames(frames, transformations):
	return transformations(frames)

def get_all_arp_frames(video_path, num_segments):
	rank_pooled_frames = run_video_appx_rank_pooling(video_path, num_segments)
	return rank_pooled_frames

def get_group_transforms(args, model):
	input_mean = model.input_mean
	input_std = model.input_std
	length = (3 if args.modality in ('RGB', 'ARP', 'RGBDiff') else 2)
	if isinstance(input_mean, int):
		input_mean = [input_mean] * length
	if isinstance(input_mean, int):
		input_mean = [input_mean] * length
	
	transforms = torchvision.transforms.Compose([
					   model.get_augmentation(),
					   Stack(roll=True),
					   ToTorchFormatTensor(div=False),
					   GroupNormalize(input_mean, input_std),
				   ])
	return transforms

def load_model_from_checkpoint(
	model,
	checkpointfile,
	was_data_paralleled=True
):
	if was_data_paralleled:
		model = torch.nn.DataParallel(model, device_ids=[0,]).cuda()
	checkpoint = torch.load(checkpointfile)
	start_epoch = checkpoint['epoch']
	best_prec1 = checkpoint['best_prec1']
	
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)
	return model

def init_model(num_classes, new_length, args):
	model = TSN(num_classes, args.num_segments, args.modality,
				base_model=args.arch,
				new_length=new_length,
				consensus_type=args.consensus_type,
				dropout=0.5,
				partial_bn=False)

	crop_size = model.crop_size
	scale_size = model.scale_size
	input_mean = model.input_mean
	input_size = model.input_size
	input_std = model.input_std
	policies = model.get_optim_policies()
	train_augmentation = model.get_augmentation()

	cropping = torchvision.transforms.Compose([
		GroupScale(scale_size),
		GroupCenterCrop(input_size),
	])
	return model

def get_num_classes(dataset):
	if dataset == 'ucf101':
		num_class = 101
	elif dataset == 'hmdb51':
		num_class = 51
	elif dataset == 'kinetics':
		num_class = 400
	elif dataset == 'saag01':
		num_class = 2
	else:
		raise ValueError('Unknown dataset ' + args.dataset)

	return num_class

def get_data_length(modality):
	if modality in ['RGB', 'ARP']:
		data_length = 1
	elif modality in ['Flow', 'RGBDiff']:
		data_length = 5

	return data_length

if __name__ == '__main__':
	args = get_options()
	video_path = args.video_path
	dataset = args.dataset
	architecture = args.arch
	checkpoint = args.checkpoint
	num_segments = args.num_segments
	modality = args.modality

	num_classes = get_num_classes(dataset)
	data_length = get_data_length(modality)

	print("--------------------------------------------------------------------------")
	print("> Model Init: %s" % args.arch)

	model = init_model(num_classes, data_length, args)

	print("--------------------------------------------------------------------------")
	print("> Loading Video to Frames: %s" % video_path)

	frames = get_all_arp_frames(video_path, num_segments)
	num_frames = len(frames)
	print(".. Frame shape: ", num_frames, frames[0].size)

	print("--------------------------------------------------------------------------")
	print("> Sampling Video Frames: sampling median of %d segments" % num_segments)
	sample_indices = generate_sample_indices(num_frames, data_length, num_segments)
	frames = sample_frames(frames, sample_indices, data_length)
	print(".. Frame shape: ", len(frames), frames[0].size)

	print("--------------------------------------------------------------------------")
	print("Transforming Frames to dataset:")
	
	transformations = get_group_transforms(args, model)
	processed_input = transforms_frames(frames, transformations)
	
	
	print(".. Transformed shape: ", processed_input.size())

	print("--------------------------------------------------------------------------")
	print("> Model Load Checkpoint: %s" % checkpoint)
	model = load_model_from_checkpoint(model, checkpoint, was_data_paralleled=True)
	
	torchsummary.summary(model, processed_input.size())

	print("--------------------------------------------------------------------------")
	print("> Prediction output: ")

	output = model(processed_input)
	_, pred = output.topk(1)

	print(".. class ", pred.item())



	



