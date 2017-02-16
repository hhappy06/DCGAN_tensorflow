import os, sys
import tensorflow as tf
import numpy as np
import time
from model.dcgan import DCGAN
from input.line_parser.line_parser import ImageParser
from input.data_reader import read_data
from loss.dcgan_loss import DCGANLoss
from train_op import d_train_opt, g_train_opt

_BATCH_SIZE_ = 64
_Z_DIM_ = 100
_DATA_DIR_ = './data/mnist/train_images'
_EPOCH_ = 10
_TRAINING_SET_SIZE_ = 60000
_CSVFILE_ = ['./data/mnist/train_images/file_list']

_OUTPUT_INFO_FREQUENCE_ = 100
_OUTPUT_IMAGE_FREQUENCE_ = 100

line_parser = ImageParser()
dcgan_loss = DCGANLoss()

def train():
	with tf.Graph().as_default():
		images, labels = read_data(_CSVFILE_, line_parser = line_parser, data_dir = _DATA_DIR_, batch_size = _BATCH_SIZE_)
		z = tf.placeholder(tf.float32, [None, _Z_DIM_], name = 'z')

		dcgan = DCGAN('model', './checkpoint')
		d_loss_logit, g_loss_logit, g_images = dcgan.inference(images, z)
		d_real_loss, d_fake_loss, d_loss, g_loss = dcgan_loss.loss(d_loss_logit, g_loss_logit)

		# summary
		sum_z = tf.summary.histogram('z', z)
		sum_g_images = tf.summary.image('g_images', g_images)
		sum_d_real_loss = tf.summary.scalar('d_real_loss', d_real_loss)
		sum_d_fake_loss = tf.summary.scalar('d_fake_loss', d_fake_loss)
		sum_d_loss = tf.summary.scalar('d_loss', d_loss)
		sum_g_loss = tf.summary.scalar('g_loss', g_loss)

		sum_g = tf.summary.merge([sum_z, sum_g_loss])
		sum_d = tf.summary.merge([sum_z, sum_d_loss, sum_d_real_loss, sum_d_fake_loss])

		# opt
		trainable_vars = tf.trainable_variables()
		d_vars = [var for var in trainable_vars if 'd_' in var.name]
		g_vars = [var for var in trainable_vars if 'g_' in var.name]

		d_opt = d_train_opt(d_loss, d_vars)
		g_opt = g_train_opt(g_loss, g_vars)

		# generate_images for showing
		generate_images = dcgan.generate_images(z, 4, 4)
		
		# initialize variable
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		session = tf.Session()
		file_writer = tf.summary.FileWriter('./logs', session.graph)
		session.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		print 'DCGAN training starts...'
		sys.stdout.flush()
		counter = 0
		max_steps = int(_TRAINING_SET_SIZE_ / _BATCH_SIZE_)
		for epoch in xrange(_EPOCH_):
			for step in xrange(max_steps):
				batch_z = np.random.uniform(-1, 1, [_BATCH_SIZE_, _Z_DIM_]).astype(np.float32)

				_, summary_str, error_d_real_loss, error_d_fake_loss, error_g_loss = session.run([d_opt, sum_d, d_real_loss, d_fake_loss, g_loss], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				_, summary_str = session.run([g_opt, sum_g], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				_, summary_str = session.run([g_opt, sum_g], feed_dict = {
					z: batch_z})
				file_writer.add_summary(summary_str, counter)

				file_writer.flush()

				counter += 1

				if counter % _OUTPUT_INFO_FREQUENCE_ == 0:
					print 'step: (%d, %d), real: %f, fake: %f, g_loss:%f'%(epoch, step, error_d_real_loss, error_d_fake_loss, error_g_loss)
					sys.stdout.flush()

				if counter % _OUTPUT_IMAGE_FREQUENCE_ == 0:
					batch_z = np.random.uniform(-1, 1, [_BATCH_SIZE_, _Z_DIM_]).astype(np.float32)
					generated_image_eval = session.run(generate_images, {z: batch_z})
					filename = os.path.join('./result', 'out_%03d_%05d.png' %(epoch, step))
					with open(filename, 'wb') as f:
						f.write(generated_image_eval)
					print 'output generated image: %s'%(filename)
					sys.stdout.flush()

		print 'training done!'
		file_writer.close()
		coord.request_stop()
		coord.join(threads)
		session.close()

if __name__ == '__main__':
	train()
