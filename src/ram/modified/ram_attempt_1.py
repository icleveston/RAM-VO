
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def Linear(inputs, out_dim, name='Linear', nl=tf.identity):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        
        inputs = batch_flatten(inputs)
        
        in_dim = inputs.get_shape().as_list()[1]
        
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        
        act = tf.nn.xw_plus_b(inputs, weights, biases)

        return nl(act, name='output')


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


class RAM:
    
    def __init__(self, config):
        
        self._config = config
        self._im_size = config.im_size
        self._n_channel = config.im_channel
        self._glimpse_num = config.glimpse_num
        self._glimpse_scale = config.glimpse_scale
        self._n_loc_sample = config.sample
        self._loc_std = config.loc_std
        self._unit_pixel = config.unit_pixel
        self._n_step = config.n_step
        self._dim_out = config.dim_out
        self._max_grad_norm = config.max_grad_norm

    def _create_train_model(self):
        self.set_is_training(True)
        self.lr = tf.placeholder(tf.float32, name='lr')
        self._create_input()
        self.layers = {}
        self.core_net(self.input_im)
    
    def _create_predict_model(self):
        self.set_is_training(False)
        self._create_input()
        self.layers = {}
        self.core_net(self.input_im)

    def _create_input(self):
        self.label = tf.placeholder(tf.float32, [None], name='label')
        self.image = tf.placeholder(tf.float32, [None, None, None, self._n_channel], name='image')

        self.input_im = self.image
        self.input_label = self.label

    def core_net(self, inputs_im):
        
        self.layers['loc_mean'] = []
        self.layers['loc_sample'] = []
        self.layers['rnn_outputs'] = []
        self.layers['retina_reprsent'] = []

        cell_size = 256
        batch_size = tf.shape(inputs_im)[0]

        init_loc_mean = tf.ones((batch_size, 2))
        loc_sample = tf.random_uniform((batch_size, 2), minval=-1, maxval=1)
        glimpse_out = self.glimpse_net(inputs_im, loc_sample)

        if self.is_training:
            inputs_im = tf.tile(inputs_im, [self._n_loc_sample, 1, 1, 1])
            glimpse_out = tf.tile(glimpse_out, [self._n_loc_sample, 1])
            batch_size = tf.shape(glimpse_out)[0]
            init_loc_mean = tf.tile(init_loc_mean, [self._n_loc_sample, 1])
            loc_sample = tf.tile(loc_sample, [self._n_loc_sample, 1])

        self.layers['loc_mean'].append(init_loc_mean)
        self.layers['loc_sample'].append(loc_sample)

        # RNN of core net
        h_prev = tf.zeros((batch_size, cell_size))
        
        for step_id in range(0, self._n_step):
            
            with tf.variable_scope('core_net'):
                h = tf.nn.relu(Linear(h_prev, cell_size, 'lh') + Linear(glimpse_out, cell_size, 'lg'), name='h')

            # core net does not trained through locatiion net
            loc_mean = self.location_net(tf.stop_gradient(h))
            
            if self.is_training:
                loc_sample = tf.stop_gradient(sample_normal_single(loc_mean, stddev=self._loc_std))
            else:
                loc_sample = tf.stop_gradient(sample_normal_single(loc_mean, stddev=self._loc_std))

            glimpse_out = self.glimpse_net(inputs_im, loc_sample)
            action = self.action_net(h)

            # do not restore the last step location
            if step_id < self._n_step - 1:
                self.layers['loc_mean'].append(loc_mean)
                self.layers['loc_sample'].append(loc_sample)
            self.layers['rnn_outputs'].append(h)

            h_prev = h

        self.layers['pred'] = action
        #self.layers['prob'] = tf.nn.softmax(logits=action, name='prob')
        #self.layers['pred'] = tf.argmax(action, axis=1)

    def glimpse_net(self, inputs, l_sample):
        """
            Args:
                inputs: [batch, h, w, c]
                l_sample: [batch, 2]
        """

        with tf.name_scope('glimpse_sensor'):
            max_r = int(self._glimpse_num * (2 ** (self._glimpse_scale - 2)))
            inputs_pad = tf.pad(
                inputs,
                [[0, 0], [max_r, max_r], [max_r, max_r], [0, 0]],
                'CONSTANT') 

            #TODO use clipped location to compute prob or not?
            l_sample = tf.clip_by_value(l_sample, -1.0, 1.0)

            l_sample_adj = l_sample * 1.0 * self._unit_pixel / (self._im_size / 2 + max_r)
            
            retina_reprsent = []
            
            for g_id in range(0, self._glimpse_scale):
                
                cur_size = self._glimpse_num * (2 ** g_id)
                
                cur_glimpse = tf.image.extract_glimpse(
                    inputs_pad,
                    size=[cur_size, cur_size],
                    offsets=l_sample_adj,
                    centered=True,
                    normalized=True,
                    uniform_noise=True,
                    name='glimpse_sensor',
                )
                
                cur_glimpse = tf.image.resize_images(
                    cur_glimpse,
                    size=[self._glimpse_num, self._glimpse_num],
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                )
                
                retina_reprsent.append(cur_glimpse)
                
            retina_reprsent = tf.concat(retina_reprsent, axis=-1)
            
            self.layers['retina_reprsent'].append(retina_reprsent)
            
        with tf.variable_scope('glimpse_net'):
            
            out_dim = 128
            hg = Linear(retina_reprsent, out_dim, name='hg', nl=tf.nn.relu)
            hl = Linear(l_sample, out_dim, name='hl', nl=tf.nn.relu)

            out_dim = 256
            g = tf.nn.relu(Linear(hl, out_dim, 'lhg') + Linear(hg, out_dim, 'lhl'), name='g')
            
            return g

    def location_net(self, core_state):
        
        with tf.variable_scope('loc_net'):
            
            l_mean = Linear(core_state, 2, name='l_mean')
            # l_mean = tf.tanh(l_mean)
            l_mean = tf.clip_by_value(l_mean, -1., 1.)
            return l_mean

    def action_net(self, core_state):
        
        with tf.variable_scope('act_net'):
            
            act = Linear(core_state, self._dim_out, name='act')
            return act
        
    def train(self, train_data, valid_data):
        
        self._create_train_model()
        
        trainer = Trainer(self, train_data, init_lr=self._config.lr)
    
        writer = tf.summary.FileWriter(self._config.output_path)
        saver = tf.train.Saver()

        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        
        with tf.Session(config=sessconfig) as sess:
            
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            
            for step in range(0, self._config.epoch):
                
                trainer.train_epoch(sess, summary_writer=writer)
                trainer.valid_epoch(sess, valid_data, self._config.batch, summary_writer=writer)
                saver.save(sess, '{}{}-{}'.format(self._config.output_path, self._config.name, self._n_step), global_step=step)
                writer.close()
                
    def predict(self, load):
        
        self._create_predict_model()
        
        predictor = Predictor(self)
        saver = tf.train.Saver()

        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        
        with tf.Session(config=sessconfig) as sess:
            
            sess.run(tf.global_variables_initializer())
            
            saver.restore(sess, '{}{}-{}'.format(self._config.output_path, self._config.name, load))
                
            im, label = valid_data.next_batch()
            
            predictor.test_batch(
                                sess,
                                im,
                                label,
                                unit_pixel=self._config.config.unit_pixel,
                                size=self._config.config.glimpse,
                                scale=self._config.config.n_scales,
                                save_path=self._config.output_path
            )
            
    def evaluate(self, load):
        
        self._create_predict_model()
        
        predictor = Predictor(self)
         
        saver = tf.train.Saver()

        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        
        with tf.Session(config=sessconfig) as sess:
            sess.run(tf.global_variables_initializer())
            
            saver.restore(sess, '{}ram-{}-mnist-step-6-{}'.format(self._config.output_path, self._config.name, load))
                
            predictor.evaluate(sess, valid_data)

    def _comp_baselines(self):
        
        with tf.variable_scope('baseline'):
            
            # core net does not trained through baseline loss
            rnn_outputs = tf.stop_gradient(self.layers['rnn_outputs'])
            baselines = []
            
            for step_id in range(0, self._n_step-1):
                b = Linear(rnn_outputs[step_id], 1, name='baseline')
                b = tf.squeeze(b, axis=-1)
                baselines.append(b)
            
            baselines = tf.stack(baselines) # [n_step, b_size]
            baselines = tf.transpose(baselines) # [b_size, n_step]
            
            return baselines

    def get_train_op(self):
        
        global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False)
        
        cur_lr = tf.train.exponential_decay(self.lr,
                                            global_step=global_step,
                                            decay_steps=500,
                                            decay_rate=0.97,
                                            staircase=True)
        
        cur_lr = tf.maximum(cur_lr, self.lr / 10.0)
        self.cur_lr = cur_lr

        loss = self.get_loss()
        var_list = tf.trainable_variables()
        grads = tf.gradients(loss, var_list)
        
        # [tf.summary.histogram('gradient/' + var.name, grad, 
        #  collections = [tf.GraphKeys.SUMMARIES])
        #  for grad, var in zip(grads, var_list)]
        grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        opt = tf.train.AdamOptimizer(cur_lr)
        
        train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)
        
        return train_op

    def _get_loss(self):
        
        return self._cls_loss() + self._REINFORCE()

    def _cls_loss(self):
        
        with tf.name_scope('class_cross_entropy'):
            
            label = self.input_label

            
            if self.is_training:
                label = tf.tile(label, [1])
                
            predictions = tf.reshape(self.layers['pred'], [-1])
            
            mse = tf.losses.mean_squared_error(labels=label, predictions=predictions)
            
            return mse

    def _REINFORCE(self):
        
        with tf.name_scope('REINFORCE'):
            
            label = self.input_label
            
            if self.is_training:
                label = tf.tile(label, [1])
                
            pred = self.layers['pred']
            
            reward = tf.stop_gradient(tf.cast(tf.equal(pred, label), tf.float32))
                        
            expanded_reward = tf.expand_dims(reward, 1)
             
                        
            reward = tf.tile(reward, [1, self._n_step - 1]) # [b_size, n_step]

            loc_mean = tf.stack(self.layers['loc_mean'][1:]) # [n_step, b_size, 2]
            loc_sample = tf.stack(self.layers['loc_sample'][1:]) # [n_step, b_size, 2]
            dist = tf.distributions.Normal(loc=loc_mean, scale=self._loc_std)
            log_prob = dist.log_prob(loc_sample) # [n_step, b_size, 2]
            log_prob = tf.reduce_sum(log_prob, -1) # [n_step, b_size]
            log_prob = tf.transpose(log_prob) # [b_size, n_step]

            baselines = self._comp_baselines()
                       
            b_mse = tf.losses.mean_squared_error(labels=reward, predictions=baselines)
            
            low_var_reward = (reward - tf.stop_gradient(baselines))
            
            REINFORCE_reward = tf.reduce_mean(log_prob * low_var_reward)

            loss = -REINFORCE_reward + b_mse
            
            return loss

    def get_loss(self):
        
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            
            return self.loss

    def get_summary(self):
        
        return tf.summary.merge_all() 

    def get_accuracy(self):
        
        label = self.input_label
        
        if self.is_training:
            label = tf.tile(label, [1])
                
        pred = self.layers['pred']
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))
        
        return accuracy

    def set_is_training(self, is_training=True):
        
        self.is_training = is_training


def sample_normal_single(mean, stddev, name=None):
	return tf.random_normal(
		# shape=mean.get_shape(),
		shape=tf.shape(mean),
    	mean=mean,
    	stddev=stddev,
    	dtype=tf.float32,
    	seed=None,
    	name=name,
    )

def get_shape2D(in_val):
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))


class Trainer:
    
    def __init__(self, model, train_data, init_lr=1e-3):
        
        self._model = model
        self._train_data = train_data
        self._lr = init_lr
        self._train_op = model.get_train_op()
        self._loss_op = model.get_loss()
        self._accuracy_op = model.get_accuracy()
        self._lr_op = model.cur_lr

        self.global_step = 0

    def train_epoch(self, sess, summary_writer=None):
        
        self._model.set_is_training(True)
        
        cur_epoch = self._train_data.epochs_completed
        step = 0
        loss_sum = 0
        acc_sum = 0
        
        while cur_epoch == self._train_data.epochs_completed:
            
            self.global_step += 1
            step += 1

            im, label = self._train_data.next_batch()
            
            _, loss, acc, cur_lr = sess.run([self._train_op,
                                             self._loss_op, 
                                             self._accuracy_op,
                                             self._lr_op], 
                                             feed_dict={self._model.image: im,
                                                    self._model.label: label,
                                                    self._model.lr: self._lr
                                    })

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                print('step: {}, loss: {:.4f}, accuracy: {:.4f}'.format(self.global_step, loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}, lr:{}'.format(cur_epoch, loss_sum * 1.0 / step, acc_sum * 1.0 / step, cur_lr))
              
        if summary_writer is not None:
            
            s = tf.Summary()
            
            s.value.add(tag='train/loss', simple_value=loss_sum * 1.0 / step)
            s.value.add(tag='train/accuracy', simple_value=acc_sum * 1.0 / step)
            
            summary_writer.add_summary(s, self.global_step)


    def valid_epoch(self, sess, dataflow, batch_size, summary_writer=None):
        
        self._model.set_is_training(False)

        step = 0
        loss_sum = 0
        acc_sum = 0
        
        while dataflow.epochs_completed == 0:
            
            print("HWRE")
            
            step += 1
            im, label = dataflow.next_batch()
            
            loss, acc = sess.run([self._loss_op, self._accuracy_op], feed_dict={
                self._model.image: im,
                self._model.label: label,
            })
                
            loss_sum += loss
            acc_sum += acc
            
        #print('valid loss: {:.4f}, accuracy: {:.4f}'.format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        #if summary_writer is not None:
            
        #    s = tf.Summary()
            
        #    s.value.add(tag='valid/loss', simple_value=loss_sum * 1.0 / step)
        #    s.value.add(tag='valid/accuracy', simple_value=acc_sum * 1.0 / step)
            
        #    summary_writer.add_summary(s, self.global_step)

        self._model.set_is_training(True)
        

class Predictor:
    
    def __init__(self, model):
        self._model = model

        self._accuracy_op = model.get_accuracy()
        self._pred_op = model.layers['pred']
        self._sample_loc_op = model.layers['loc_sample']

    def evaluate(self, sess, dataflow, batch_size=None):
        self._model.set_is_training(False)

        step = 0
        acc_sum = 0
        while dataflow.epochs_completed == 0:
            step += 1
            
            im, label= dataflow.next_batch()
            
            acc = sess.run(
                self._accuracy_op, 
                feed_dict={self._model.image: im,
                           self._model.label: label,
                           })
                
            acc_sum += acc
            
        print('accuracy: {:.4f}'.format(acc_sum * 1.0 / step))

        self._model.set_is_training(True)

    def test_batch(self, sess, x, y, unit_pixel, size, scale, save_path=''):
        def draw_bbx(ax, x, y):
            rect = patches.Rectangle(
                (x, y), cur_size, cur_size, edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        self._model.set_is_training(False)
        
        loc_list, pred, input_im, glimpses = sess.run(
            [self._sample_loc_op, self._pred_op, self._model.input_im,
             self._model.layers['retina_reprsent']],
            feed_dict={self._model.image: x,
                       self._model.label: y,
                        })

        pad_r = size * (2 ** (scale - 2))
        print(pad_r)
        im_size = input_im[0].shape[0]
        loc_list = np.clip(np.array(loc_list), -1.0, 1.0)
        loc_list = loc_list * 1.0 * unit_pixel / (im_size / 2 + pad_r)
        loc_list = (loc_list + 1.0) * 1.0 / 2 * (im_size + pad_r * 2)
        offset = pad_r

        print(pred)
        for step_id, cur_loc in enumerate(loc_list):
            im_id = 0
            glimpse = glimpses[step_id]
            for im, loc, cur_glimpse in zip(input_im, cur_loc, glimpse):
                im_id += 1                
                fig, ax = plt.subplots(1)
                ax.imshow(np.squeeze(im), cmap='gray')
                for scale_id in range(0, scale):
                    cur_size = size * 2 ** scale_id
                    side = cur_size * 1.0 / 2
                    x = loc[1] - side - offset
                    y = loc[0] - side - offset
                    draw_bbx(ax, x, y)
                # plt.show()
                for i in range(0, scale):
                    scipy.misc.imsave(
                        os.path.join(save_path,'im_{}_glimpse_{}_step_{}.png').format(im_id, i, step_id),
                        np.squeeze(cur_glimpse[:,:,i]))
                plt.savefig(os.path.join(
                    save_path,'im_{}_step_{}.png').format(im_id, step_id))
                plt.close(fig)

        self._model.set_is_training(True)
