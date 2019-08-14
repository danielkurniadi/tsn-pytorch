import os
import click
from click import echo

from multiprocessing import current_process

# Computation libs
import numpy as np
import cv2

from multiprocessing import Pool, current_process

# File utilities
from utils import (
	safe_mkdir,
	search_files_recursively,
	get_basename,
	clean_filename
)


class Buffer():
	def __init__(self, size):
		self.size = max(int(size), 1)
		self.container = []

	def enqueue(self, item):
		if len(self.container) < self.size:
			self.container.append(item)
		else:
			print('Buffer full')

	def dequeue(self):
		if not self.isempty():
			self.container.pop(0)
		else:
			print("Buffer empty")

	def clear(self):
		container = self.container
		self.container = []
		return np.array(container)

	def get(self):
		return np.array(self.container)

	def isempty(self):
		return len(self.container) == 0

	def isfull(self):
		return (len(self.container) == self.size)


def cvApproxRankPooling(imgs):
	T = len(imgs)
  
	harmonics = []
	harmonic = 0
	for t in range(0, T+1):
		harmonics.append(harmonic)
		harmonic += float(1)/(t+1)

	weights = []
	for t in range(1 ,T+1):
		weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
		weights.append(weight)
		
	feature_vectors = []
	for i in range(len(weights)):
		feature_vectors.append(imgs[i] * weights[i])

	feature_vectors = np.array(feature_vectors)

	rank_pooled = np.sum(feature_vectors, axis=0)
	rank_pooled = cv2.normalize(rank_pooled, None, alpha=0, beta=255, 
		norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	return rank_pooled


def run_video_appx_rank_pooling(
	video_path,
	outdir,
	img_ext='.jpg',
	buffer_size=24,
):
	"""Approximated Rank Pooling (ARP) runner for video input

	Outputs Rank pooled frames from a video.
	"""
	def _run_appx_rank_pooling(frames, outpath):
		rank_pooled = cvApproxRankPooling(frames)

		cv2.imwrite(outpath, rank_pooled)

	arp_name_tmpl = 'arp_{:05d}' + img_ext
	safe_mkdir(outdir)  # create directory for each video data

	current = current_process()
	cap = cv2.VideoCapture(video_path)
	buffer = Buffer(buffer_size)
	success = True

	count = 1
	while success:
		success, frame = cap.read()

		if buffer.isfull():
			frames = buffer.clear()

			arp_name = arp_name_tmpl.format(count)
			arp_outpath = os.path.join(outdir, arp_name)

			_run_appx_rank_pooling(frames, arp_outpath)
			count += 1

		buffer.enqueue(frame)

	cap.release()

	print(".. Finished running appx rankpool to %s" % outdir)


def video_appxRankPooling(
	source,
	dest,
	n_jobs,
	buffer_size,
	img_ext
):
	print(". Executing appx_rank_pool on video...")
	safe_mkdir(dest)
	
	for class_folder in os.listdir(source):     # run appx rank pool for each video in all class_folder
		video_files = search_files_recursively(
			os.path.join(source, class_folder)
		)
		outfolder = os.path.join(dest, class_folder)

		safe_mkdir(outfolder)

		# take only the basename of each video url, clean name from dot and whitespace
		# and use this basename for output image name
		outdir = [
			os.path.join(outfolder, clean_filename(get_basename(video_file)))
			for video_file in video_files
		]
		img_exts = [img_ext]* len(outdir)  # TODO: optimise this extension duplicating given every element is constant
		buffer_sizes = [buffer_size] * len(outdir)

		print(". Current class folder: %s, total:%d" %(class_folder, len(video_files)))

		run_args = list(zip(video_files, outdir, img_exts, buffer_sizes))
		results = Pool(n_jobs).starmap(
			run_video_appx_rank_pooling, run_args
		)

		print(". Finished %s." % class_folder)


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
	parser.add_argument('source', type=str)
	parser.add_argument('dest', type=str)
	parser.add_argument('-j', '--n_jobs', type=int, default=5)
	parser.add_argument('-b', '--buffer_size', type=int, default=24)
	parser.add_argument('--img_ext', type=str, default='.jpg')
	args = parser.parse_args()
	
	video_appxRankPooling(
		args.source,
		args.dest,
		args.n_jobs,
		args.buffer_size,
		args.img_ext
	)
	
