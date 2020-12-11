from os import listdir, path
import numpy as np
import scipy
import cv2
import os, sys
from discriminator2 import Discriminator_Model
import torch
from generator import Generator_Model
from tqdm import tqdm
from glob import glob
import pickle, argparse

half_window_size = 4
mel_step_size = 27

def frame_id(fname):
	return int(os.path.basename(fname).split('.')[0])

def choose_ip_frame(frames, gt_frame):
	selected_frames = [f for f in frames if np.abs(frame_id(gt_frame) - frame_id(f)) >= 6]
	return np.random.choice(selected_frames)

def get_audio_segment(center_frame, spec):
	center_frame_id = frame_id(center_frame)
	start_frame_id = center_frame_id - half_window_size

	start_idx = int((80.0/25.0) * start_frame_id) # 25 is fps of LRS2
	end_idx = start_idx + mel_step_size

	return spec[:, start_idx : end_idx] if end_idx <= spec.shape[1] else None

def datagen(args):
	all_images = args.all_images
	batch_size = args.batch_size

	while(1):
		np.random.shuffle(all_images)
		batches = [all_images[i:i + args.batch_size] for i in range(0, len(all_images), args.batch_size)]

		for batch in batches:
			img_gt_batch = []
			img_ip_batch = []
			mel_batch = []
			
			for img_name in batch:
				gt_fname = os.path.basename(img_name)
				dir_name = img_name.replace(gt_fname, '')
				frames = glob(dir_name + '/*.jpg')
				if len(frames) < 12:
					continue

				mel_fname = dir_name + '/mels.npz'
				try:
					mel = np.load(mel_fname)['spec']
				except:
					continue

				mel = get_audio_segment(gt_fname, mel)

				if mel is None or mel.shape[1] != mel_step_size:
					continue

				if sum(np.isnan(mel.flatten())) > 0:
					continue
				
				img_gt = cv2.imread(img_name)
				img_gt = cv2.resize(img_gt, (args.img_size, args.img_size))
				
				ip_fname = choose_ip_frame(frames, gt_fname)
				img_ip = cv2.imread(os.path.join(dir_name, ip_fname))
				img_ip = cv2.resize(img_ip, (args.img_size, args.img_size))
				
				img_gt_batch.append(img_gt)
				img_ip_batch.append(img_ip)
				mel_batch.append(mel)

			img_gt_batch = np.asarray(img_gt_batch)
			img_ip_batch = np.asarray(img_ip_batch)
			mel_batch = np.expand_dims(np.asarray(mel_batch), 3)
			
			img_gt_batch_masked = img_gt_batch.copy()
			img_gt_batch_masked[:, args.img_size//2:,...] = 0.
			img_ip_batch = np.concatenate([img_ip_batch, img_gt_batch_masked], axis=3)
			
			yield [img_ip_batch/255.0, mel_batch], img_gt_batch/255.0

parser = argparse.ArgumentParser(description='Pytorch implementation of LipGAN')

parser.add_argument('--data_root', type=str, help='LRS3 preprocessed dataset root to train on', required=True)
parser.add_argument('--logdir', type=str, help='Folder to store checkpoints & generated images', default='logs/')

parser.add_argument('--model', type=str, help='Model name to use: basic|residual', default='residual')
parser.add_argument('--resume_gen', help='Path to weight file to load into the generator', default=None)
parser.add_argument('--resume_disc', help='Path to weight file to load into the discriminator', default=None)
parser.add_argument('--checkpoint_freq', type=int, help='Frequency of checkpointing', default=1000)

parser.add_argument('--n_gpu', type=int, help='Number of GPUs to use', default=1)
parser.add_argument('--batch_size', type=int, help='Single GPU batch size', default=96)
parser.add_argument('--lr', type=float, help='Initial learning rate', default=1e-4)
parser.add_argument('--img_size', type=int, help='Size of input image', default=96)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=20000)

parser.add_argument('--all_images', default='filenames.pkl', help='Filename for caching image paths')
args = parser.parse_args()

all_images = glob(path.join("{}/*/*/*.jpg".format(args.data_root)))
pickle.dump(all_images, open(path.join(args.logdir, args.all_images), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
args.all_images = all_images

print ("Will be training on {} images".format(len(args.all_images)))

# Initialize the generator and the Discriminator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = Generator_Model()
disc = Discriminator_Model() # Pytorch

gen = gen.to(device)
disc = disc.to(device)

# Use an already-created generator or discriminator
if args.resume_gen:
	gen.load_state_dict(torch.load(args.resume_gen))
	print('Resuming generator from : {}'.format(args.resume_gen))
if args.resume_disc:
	disc.load_state_dict(torch.load(args.resume_dicc))
	print('Resuming discriminator from : {}'.format(args.resume_disc))

args.batch_size = args.n_gpu * args.batch_size
train_datagen = datagen(args)

for e in range(args.epochs):
	prog_bar = tqdm(range(len(args.all_images) // args.batch_size))
	disc_loss, unsync_loss, sync_loss, gen_loss_mae, gen_loss_adv = 0., 0., 0., 0., 0.
	prog_bar.set_description('Starting epoch {}'.format(e))
	for batch_idx in prog_bar:
		gen.train()
		disc.train()

		(dummy_faces, audio), real_faces = next(train_datagen)
		real = np.zeros((len(real_faces), 1))
		fake = np.ones((len(real_faces), 1))

		gen_fakes = gen(audio, dummy_faces) # predict fake
		
		print(gen_fakes)

### TODO: Adjust / convert everything below this line
		### Train Discriminator
		disc.optimizer.zero_grad()
		gen.optimizer.zero_grad()

		# Replaced: disc_loss += disc.train_on_batch...
		d_output = disc(gen_fakes, mel_step_size, fake)
		d_loss += disc.contrastive_loss(fake, d_output)

		# Replaced: disc_loss += disc.test_on_batch...
		disc.eval()
		with torch.nograd:
			d_test_out = disc(real_faces, mel_step_size, fake)
			d_unsync_loss = disc.contrastive_loss(fake, d_test_out)
		disc.train()

		# Replace: disc.train_on_batch([real_faces, audio], real)
		d_sync_out = disc(real_faces, mel_step_size, real)
		d_sync_loss += disc.contrastive_loss(real, d_sync_out)

		# train generator 
		gen.train()
		gen.optimizer.step()

		if (batch_idx + 1) % (args.checkpoint_freq // 10) == 0:
			if (batch_idx + 1) % args.checkpoint_freq == 0:
				disc.save(path.join(args.logdir, 'disc.h5'))
				gen.save(path.join(args.logdir, 'gen.h5'))
	
			collage = np.concatenate([dummy_faces[...,:3], real_faces, gen_fakes], axis=2)
			collage *= 255.
			collage = np.clip(collage, 0., 255.).astype(np.uint8)
			
			for i in range(len(collage)):
				cv2.imwrite(path.join(args.logdir, 'gen_faces/{}.jpg'.format(i)), collage[i])
