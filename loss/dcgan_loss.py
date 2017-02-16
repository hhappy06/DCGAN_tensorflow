from __future__ import absolute_import
import tensorflow as tf

class DCGANLoss:
	def loss(self, d_logit, g_logit):
		d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logit, tf.ones_like(d_logit)))
		d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(g_logit, tf.zeros_like(g_logit)))
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(g_logit, tf.ones_like(g_logit)))
		d_loss = d_real_loss + d_fake_loss

		return d_real_loss, d_fake_loss, d_loss, g_loss