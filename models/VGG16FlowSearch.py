import copy
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import vgg
from tf_models.research.slim.preprocessing import vgg_preprocessing


class VGG16FlowSearch():
    def __init__(self):
        self.graph = self.build_graph(tf.Graph())

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)

        self.init()

    def build_graph(self, graph):
        with graph.as_default():
            self.input1 = tf.placeholder(tf.uint8, (None, None, 3), name='image1')
            self.input2 = tf.placeholder(tf.uint8, (None, None, 3), name='image2')
            _, endpoints1 = vgg.vgg_16(self.preprocess_input(self.input1), is_training=False)
            tf.get_variable_scope().reuse_variables()
            _, endpoints2 = vgg.vgg_16(self.preprocess_input(self.input2), is_training=False)
            self.init_fn = self.get_vgg_init_fn('data/vgg_16.ckpt')

            layers1 = self.remove_layers(endpoints1)
            layers2 = self.remove_layers(endpoints2)

            L = len(layers1) - 1
            layer_names = list(layers1.keys())
            self.values1 = list(layers1.values())
            self.values2 = list(layers2.values())
            U = [None] * (L + 1)
            U_maps = [None] * (L + 1)
            print(layer_names)

            self.d_range = tf.placeholder(tf.int32, (2, 2), name='d_range')

            k_l = self.get_all_k_l(layers1, self.d_range)

            # Initialize last layer
            ds = self.get_all_d(k_l[L])

            original1 = self.values1[L]
            original2 = self.values2[L]

            def init_U(d):
                translations = tf.cast(tf.reverse_v2(d, [0]), dtype=tf.float32)
                shifted_original2 = tf.contrib.image.translate(original2, translations)
                m = tf.minimum(original1, shifted_original2) / tf.maximum(original1, shifted_original2)
                # if w and v are zero, we get nan, but we should get 0.
                m = tf.where(tf.is_nan(m), tf.zeros_like(m), m)
                return tf.reshape(m, tf.shape(m)[1:])

            U[L] = tf.map_fn(init_U, ds, dtype=tf.float32)
            U_maps[L] = ds

            for layer_id in reversed(range(0, L)):
                print('layer:', layer_id + 1, layer_names[layer_id + 1], self.values1[max(0, layer_id)].shape)
                layer_type = self.get_layer_type(layers1, layer_id + 1)
                ds = self.get_all_d(k_l[layer_id])

                original1 = self.values1[layer_id]
                original2 = self.values2[layer_id]
                forward_U = U[layer_id + 1]
                forward_U_map = U_maps[layer_id + 1]
                if layer_type == 'conv':
                    S = tf.nn.conv2d(forward_U, tf.ones([3, 3, tf.shape(forward_U)[3], 1]), [1, 1, 1, 1],
                                     'SAME')  # tf.shape(original1)[3]

                    def back_propagate(d):
                        di = tf.squeeze(tf.where(tf.reduce_all(tf.equal(forward_U_map, d), axis=1)))
                        translation = d  # tf.stack([tf.constant(0), d], axis=0)
                        result = self.propagate_convolved(original1, original2, tf.expand_dims(S[di, ...], axis=0),
                                                     translation)
                        return tf.reshape(result, tf.shape(result)[1:])
                else:  # 'pool'
                    pooled1 = self.values1[layer_id + 1]
                    pooled2 = self.values2[layer_id + 1]
                    image_size = tf.shape(original1)
                    upscaled1 = self.upscale_pooled(pooled1, image_size)
                    upscaled2 = self.upscale_pooled(pooled2, image_size)
                    equal1 = tf.cast(tf.equal(upscaled1, original1), tf.float32)
                    equal2 = tf.cast(tf.equal(original2, upscaled2), tf.float32)

                    S = self.upscale_pooled(forward_U, image_size)

                    def back_propagate(d):
                        di = tf.squeeze(tf.where(tf.reduce_all(tf.equal(forward_U_map, d // 2), axis=1)))
                        translation = d
                        result = self.propagate_pooled(equal1, equal2, tf.expand_dims(S[di, ...], axis=0), translation)
                        return tf.reshape(result, tf.shape(result)[1:])

                U[layer_id] = tf.map_fn(back_propagate, ds, dtype=tf.float32)
                U_maps[layer_id] = ds

            U_red = tf.reduce_sum(U[0], axis=3)
            likely_indices = tf.argmax(U_red, axis=0, output_type=tf.int32)
            self.U_max = tf.reduce_max(U_red, axis=0)
            self.shift = tf.gather_nd(ds, tf.expand_dims(likely_indices, axis=-1))
        return graph


    def preprocess_input(self, image):
        image = tf.to_float(image)
        image = vgg_preprocessing._mean_image_subtraction(image,
                                                          [vgg_preprocessing._R_MEAN, vgg_preprocessing._G_MEAN,
                                                           vgg_preprocessing._B_MEAN])
        image = tf.expand_dims(image, axis=0)
        return image

    def get_vgg_init_fn(self, ckpt_path):
        def not_fc(tensor):
            return not tensor.name.split('/')[1].startswith('fc')
        # Only restore layers up to the fully connect
        variables_to_restore = [v for v in tf.contrib.framework.get_variables_to_restore()
                                if not_fc(v)]
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            ckpt_path, variables_to_restore)
        return init_fn

    def remove_layers(self, layers):
        names = list(layers.keys())
        first = names.index('vgg_16/conv1/conv1_1')
        # last = names.index('vgg_16/conv3/conv3_2')
        last = names.index('vgg_16/conv5/conv5_2')
        clean = layers.copy()
        for i in set(range(len(layers))) - set(range(first, last + 1)):
            del clean[names[i]]
        return clean

    def get_all_k_l(self, layers, d_range):
        # Calculate all k_l(D)
        k_l = []
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            k_l.append(d_range)
            if self.get_layer_type(layers, i) == 'pool':
                d_range = (d_range + tf.constant([[0, 1]])) // 2
        k_l.append(d_range)
        return k_l

    def get_layer_type(self, layers, layer_id):
        layer_name = list(layers.keys())[layer_id]
        if 'pool' in layer_name:
            return 'pool'
        return 'conv'

    def get_all_d(self, kl):
        a = tf.range(kl[0, 0], kl[0, 1])
        b = tf.range(kl[1, 0], kl[1, 1])
        tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])
        tile_a = tf.expand_dims(tile_a, 2)
        tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1])
        tile_b = tf.expand_dims(tile_b, 2)
        cartesian_product = tf.reshape(tf.concat([tile_a, tile_b], axis=-1), [-1, 2])
        return cartesian_product

    def upscale_pooled(self, pooled, unpooled_size):
        " Upscale a pooled output "
        # We thought the resulting size would be unpooled_size or one more, but it
        # is actually one less because VGG uses the 'VALID' padding scheme in the
        # pooling layers
        upscaled = tf.image.resize_images(pooled, tf.shape(pooled)[1:3] * 2,
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        upscaled = tf.pad(upscaled, [[0, 0], [0, 1], [0, 1], [0, 0]],
                          'CONSTANT', constant_values=-1)
        # If unpooled_size is even, or if the pooling operation was 'SAME' rather than
        # 'VALID', we have to trim the result.
        upscaled = upscaled[:, :unpooled_size[1], :unpooled_size[2], :]
        return upscaled

    def propagate_pooled(self, equal1, equal2, S, d):

        translations = tf.cast(tf.reverse_v2(d, [0]), dtype=tf.float32)
        shifted_equal2 = tf.contrib.image.translate(equal2, translations)

        m = equal1 * shifted_equal2
        return S * m

    def propagate_convolved(self, original1, original2, S, d):

        translations = tf.cast(tf.reverse_v2(d, [0]), dtype=tf.float32)
        shifted_original2 = tf.contrib.image.translate(original2, translations)
        m = tf.minimum(original1, shifted_original2) / tf.maximum(original1, shifted_original2)
        # if w and v are zero, we get nan, but we should get 0.
        m = tf.where(tf.is_nan(m), tf.zeros_like(m), m)

        return S * m

    def infer(self, im1, im2, d_range = [[-64, 64], [-64, 64]], step = [16, 16]):

        best_shift = np.zeros((im1.shape[0], im1.shape[1], 2))
        best_umax = np.zeros(im1.shape[:2])

        with self.sess as sess:
            v = sess.run(self.values1 + self.values2, feed_dict={self.input1: im1, self.input2: im2})
            for i in range(d_range[0][0], d_range[0][1], step[0]):
                for j in range(d_range[1][0], d_range[1][1], step[1]):
                    print('Currently processing:', (i, j))
                    f_dict = {a: b for a, b in
                              zip(self.values1 + self.values2 + [self.d_range], v + [np.array([[i, i + step[0]], [j, j + step[1]]])])}
                    curr_shift, umax = sess.run([self.shift, self.U_max], feed_dict=f_dict)
                    best_shift = np.where(np.expand_dims(umax > best_umax, -1), curr_shift, best_shift)
                    best_umax = np.maximum(umax, best_umax)

        return best_shift


    def init(self):
        init_fn = self.init_fn
        init_fn(self.sess)