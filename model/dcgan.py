from __future__ import absolute_import
import numpy as np
import tensorflow as tf

_BATCH_SIZE_ = 64

_Z_DIM = 100
_y_DIM_ = None
_GF_DIM_ = 64
_DF_DIM_ = 64
_GFC_DIM_ = 1024
_DFC_DIM_ = 1024
_C_DIM_ = 1

# convolution/pool stride
_CONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_DECONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_REGULAR_FACTOR_ = 1.0e-4

def _construct_conv_layer(input_layer, output_dim, kernel_size = 5, stddev = 0.02, name = 'conv2d'):
	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [kernel_size, kernel_size, input_layer.get_shape()[-1], output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES_, padding = 'SAME')
		conv = tf.nn.bias_add(conv, bias)
		return conv

def _construct_max_pool_layer(input_layer, name = 'pool'):
	with tf.variable_scope(name):
		return tf.nn.max_pool(input_layer, ksize = _MAX_POOL_KSIZE_, strides = _MAX_POOL_STRIDES_, padding = 'SAME', name = name)

def _construct_deconv_layer(input_layer, output_shape, kernel_size = 5, stddev = 0.02, name = 'deconv'):
	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [kernel_size, kernel_size, output_shape[-1], input_layer.get_shape()[-1]]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_shape[-1]],
			initializer = tf.constant_initializer(0.0))
		deconv = tf.nn.conv2d_transpose(input_layer, weight, output_shape, strides = _DECONV_KERNEL_STRIDES_, padding = 'SAME')
		deconv = tf.nn.bias_add(deconv, bias)
		return deconv

def _construct_lrelu(input_layer, leak = 0.2, name = 'lrelu'):
	with tf.variable_scope(name):
		alpha1 = 0.5 * (1 + leak)
		alpha2 = 0.5 * (1 - leak)
		return alpha1 * input_layer + alpha2 * abs(input_layer)

def _construct_full_connection_layer(input_layer, output_dim, stddev = 0.02, name = 'fc'):
	# calculate input_layer dimension and reshape to batch * dimension
	input_dimension = 1
	for dim in input_layer.get_shape().as_list()[1:]:
		input_dimension *= dim

	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [input_dimension, output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		input_layer_reshape = tf.reshape(input_layer, [-1, input_dimension])
		fc = tf.matmul(input_layer_reshape, weight)
		tc = tf.nn.bias_add(fc, bias)
		return fc

# class _BatchNormalization:
# 	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
# 		with tf.variable_scope(name):
# 			self.epsilon = epsilon
# 			self.momentum = momentum
# 			self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
# 			self.name = name

# 	def __call__(self, input_layer, train = True):
# 		with tf.variable_scope(self.name):
# 			if train:
# 				self.beta = tf.get_variable(
# 					name = self.name + 'beta',
# 					shape = [input_layer.get_shape().as_list()[-1]],
# 					initializer = tf.constant_initializer(0.0))
# 				self.gamma = tf.get_variable(
# 					name = self.name + 'gamma',
# 					shape = [input_layer.get_shape().as_list()[-1]],
# 					initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.02))
# 				batch_mean, batch_var = tf.nn.moments(input_layer, [0, 1, 2], name='moments')
# 				ema_apply_op = self.ema.apply([batch_mean, batch_var])
# 				self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
# 				with tf.control_dependencies([ema_apply_op]):
# 					mean, var = tf.identity(batch_mean), tf.identity(batch_var)
# 			else:
# 				mean, var = self.ema_mean, self.ema_var
# 			bn = tf.nn.batch_norm_with_global_normalization(
# 				input_layer, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
# 		return bn

class _BatchNormalization:
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
				decay=self.momentum, 
				updates_collections=None,
				epsilon=self.epsilon,
				scale=True,
				is_training=train,
				scope=self.name)

# define DCGAN network
# define disctriminative network
class Discriminative:
	def __init__(self):
		self.d_bn1 = _BatchNormalization(name = 'd_bn1')
		self.d_bn2 = _BatchNormalization(name = 'd_bn2')
		self.d_bn3 = _BatchNormalization(name = 'd_bn3')

	def inference(self, images, reuse = False):
		with tf.variable_scope('discriminator') as scope:
			if reuse:
				scope.reuse_variables()

			print '='*20
			print 'discriminator input image:', images.get_shape()

			hidden0 = _construct_lrelu(_construct_conv_layer(images, _DF_DIM_, name = 'd_conv_hidden0'))
			hidden1 = _construct_lrelu(self.d_bn1(_construct_conv_layer(hidden0, _DF_DIM_ * 2, name = 'd_conv_hidden1')))
			hidden2 = _construct_lrelu(self.d_bn2(_construct_conv_layer(hidden1, _DF_DIM_ * 4, name = 'd_conv_hidden2')))
			hidden3 = _construct_lrelu(self.d_bn3(_construct_conv_layer(hidden2, _DF_DIM_ * 8, name = 'd_conv_hidden3')))

			print '='*20
			print 'discriminator ouput :', hidden3.get_shape()

			ouput = _construct_full_connection_layer(hidden3, 1, name = 'd_fc_hidden4')
			return tf.nn.sigmoid(ouput), ouput

class Generative:
	def __init__(self):
		self.g_bn0 = _BatchNormalization(name = 'g_bn0')
		self.g_bn1 = _BatchNormalization(name = 'g_bn1')
		self.g_bn2 = _BatchNormalization(name = 'g_bn2')
		self.g_bn3 = _BatchNormalization(name = 'g_bn3')

	def inference(self, z, reuse = False):
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()

			print '='*20
			print 'generator input z:', z.get_shape()

			batch_size = 64
			fc = _construct_full_connection_layer(z, _GF_DIM_ * 8 * 4 * 4, name = 'g_fc_hidden0')
			fc_reshape = tf.reshape(fc, [-1, 4, 4, _GF_DIM_ * 8])
			fc_reshape_actvie = tf.nn.relu(self.g_bn0(fc_reshape))
			deconv0 = _construct_deconv_layer(fc_reshape_actvie, [batch_size, 8, 8, _GF_DIM_ * 4], name = 'g_deconv_hidden1')
			deconv0 = tf.nn.relu(self.g_bn1(deconv0))
			deconv1 = _construct_deconv_layer(deconv0, [batch_size, 16, 16, _GF_DIM_ * 2], name = 'g_deconv_hidden2')
			deconv1 = tf.nn.relu(self.g_bn2(deconv1))
			deconv2 = _construct_deconv_layer(deconv1, [batch_size, 32, 32, _GF_DIM_ * 1], name = 'g_deconv_hidden3')
			deconv2 = tf.nn.relu(self.g_bn3(deconv2))
			deconv3 = _construct_deconv_layer(deconv2, [batch_size, 64, 64, _C_DIM_], name = 'g_deconv_hidden4')

			print "="*20
			print "generator output deconv3:", deconv3.get_shape()

			return tf.nn.tanh(deconv3)

class DCGAN:
	def __init__(self, dataset_name, checkpoint_dir):
		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir

	def inference(self, images, z):
		# generative
		print "="*100
		print 'DCGAN generative'
		self.generator = Generative()
		self.g_images = self.generator.inference(z)

		# discriminative
		print 'DCGAN discriminative'
		self.discriminator = Discriminative()
		self.d_output, self.d_output_logit = self.discriminator.inference(images)
		self.g_output, self.g_output_logit = self.discriminator.inference(self.g_images, reuse = True)

		return self.d_output_logit, self.g_output_logit, self.g_images

	def generate_images(self, z, row=8, col=8):
		images = tf.cast(tf.mul(tf.add(self.generator.inference(z, reuse = True), 1.0), 127.5), tf.uint8)
		images = [image for image in tf.split(0, _BATCH_SIZE_, images)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
		image = tf.concat(1, rows)
		return tf.image.encode_png(tf.squeeze(image, [0]))
