import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def norm(tensor):#normalzie last line
    return tensor/(tf.sqrt(tf.reduce_sum(tf.square(tensor),-1,keep_dims=True))+1e-12)
def cos(tensor1,tensor2):#by last dimension
    return tf.reduce_sum(tf.mul(norm(tensor1),norm(tensor2)),axis=-1)
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
class Model():
    def __init__(self,d_max_length=100,q_max_length=27,a_max_length=27,rnn_size=64,embedding_size=300,num_symbol=10000,sim='nn',layer=2):
        tf.reset_default_graph()
        self.d_max_length = d_max_length
        self.q_max_length = q_max_length
        self.a_max_length = a_max_length
        self.rnn_size = rnn_size
        self.lr = 1e-2
        self.input_net = {}
        self.output_net = {}
        self.cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*layer,state_is_tuple=True)
        self.cell2 = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        self.embedding_size = embedding_size
        self.num_symbol = num_symbol
        self.output_net['loss'] = []
        self.output_net['test_loss'] = []
        self.sim = sim
        self.sess = tf.Session()
        #self.batch_size = 32
    def build_model(self,):
        self.input_net['d'] = tf.placeholder(tf.int32,[None,self.d_max_length])
        self.input_net['q'] = tf.placeholder(tf.int32,[None,self.q_max_length])
        self.input_net['a'] = tf.placeholder(tf.int32,[None,self.a_max_length])
        self.input_net['d_mask'] = tf.placeholder(tf.int32,[None])
        self.input_net['q_mask'] = tf.placeholder(tf.int32,[None])
        self.input_net['a_mask'] = tf.placeholder(tf.int32,[None])
        self.W = tf.Variable(tf.random_uniform([self.num_symbol,self.embedding_size],-1.0,1.0),name="W")
        with tf.variable_scope('reader'):#document
            d_rep, d_enc_state = tf.nn.dynamic_rnn(self.cell,tf.nn.embedding_lookup(self.W,self.input_net['d']),sequence_length=self.input_net['d_mask']-1,dtype=tf.float32)
        with tf.variable_scope('reader',reuse=True):#question
            q_rep, q_enc_state = tf.nn.dynamic_rnn(self.cell,tf.nn.embedding_lookup(self.W,self.input_net['q']),sequence_length=self.input_net['q_mask']-1,dtype=tf.float32)
        last_q = last_relevant(q_rep,self.input_net['q_mask']-1)
        if self.sim == 'nn':
            w1 = tf.Variable(tf.random_uniform([self.rnn_size*2,self.rnn_size],-1.0,1.0))
            b1 = tf.Variable(tf.random_uniform([1,self.rnn_size],-1.0,1.0))
            w2 = tf.Variable(tf.random_uniform([self.rnn_size,1],-1.,1.))
            b2 = tf.Variable(tf.random_uniform([1,1],-1.,1.))
            vec = tf.reshape(tf.concat(2,[d_rep,tf.reshape(tf.tile(last_q,[1,self.d_max_length]),[-1,self.d_max_length,self.rnn_size] ) ]) ,[-1,self.rnn_size*2])
            attention_weight = tf.reshape(tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(vec,w1) + b1),w2 ) +b2),[-1,self.d_max_length,1])
            attention_sum = tf.mul(d_rep,attention_weight)
        with tf.variable_scope('reader2'):
            d_encoder_output, d_enc_state = tf.nn.dynamic_rnn(self.cell2,attention_sum,sequence_length=self.input_net['d_mask']-1,dtype=tf.float32)
        batch_size = tf.shape(self.input_net['a'])[0]
        pad = tf.constant(3,shape=[1,1])
        pad = tf.tile(pad,tf.pack([batch_size,1]))
        self.decode_input = tf.unpack(tf.concat(1,[pad,self.input_net['a']]),axis=1) # pad with "GO" symbol
        a_mask_list = tf.unpack(tf.sequence_mask(self.input_net['a_mask'],self.a_max_length,dtype=tf.float32),axis=1)
        top_states = [tf.nn.array_ops.reshape(e, [-1, 1, self.cell2.output_size]) for e in tf.unpack(d_encoder_output,axis=1)]
        attention_states = tf.nn.array_ops.concat(1, top_states)

        #self.cell2 = tf.nn.rnn_cell.OutputProjectionWrapper(self.cell2,self.num_symbol)
        if True:
            w_t = tf.get_variable("proj_w", [self.num_symbol, self.rnn_size], dtype=tf.float32)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.num_symbol])
            self.output_projection = (w, b)
            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       512, self.num_symbol),tf.float32)
            softmax_loss_function = sampled_loss
        with tf.variable_scope('decoder'):
            self.a_out, _ = tf.nn.seq2seq.embedding_attention_decoder(self.decode_input,d_enc_state,attention_states,self.cell2,self.num_symbol,self.embedding_size,output_projection=self.output_projection)
        with tf.variable_scope('decoder',reuse=True):
            self.a_pred, _ = tf.nn.seq2seq.embedding_attention_decoder(self.decode_input,d_enc_state,attention_states,self.cell2,self.num_symbol,self.embedding_size,feed_previous=True,output_projection=self.output_projection)
        for i in range(len(self.a_out)-1):
            self.output_net['loss'].append(tf.nn.seq2seq.sequence_loss_by_example([self.a_out[i]],[self.decode_input[i+1]],[a_mask_list[i]],softmax_loss_function= softmax_loss_function))
            self.output_net['test_loss'].append(tf.nn.seq2seq.sequence_loss_by_example([self.a_pred[i]],[self.decode_input[i+1]],[a_mask_list[i]],softmax_loss_function= softmax_loss_function))
            #self.output_net['loss'] += tf.nn.sparse_softmax_cross_entropy_with_logits(a_out[i],self.decode_input[i+1]) *a_mask_list[i]
            #self.output_net['test_loss'] += tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_pred[i],self.decode_input[i+1]) * a_mask_list[i]
        self.output_net['loss'] = tf.reduce_sum(self.output_net['loss'])
        self.output_net['loss'] /= tf.cast(tf.reduce_sum(self.input_net['a_mask']),tf.float32)
        self.output_net['test_loss'] = tf.reduce_sum(self.output_net['test_loss'])
        self.output_net['test_loss'] /= tf.cast(tf.reduce_sum(self.input_net['a_mask']),tf.float32)
        #with tf.variable_scope('decoder'):
        #    q_out, _ = tf.nn.seq2seq.embedding_rnn_decoder(decode_input,q_enc_state,self.cell,self.num_symbol,self.embedding_size,feed_previous=False)
        #with tf.variable_scope('decoder',reuse=True):#prediction
        #    self.q_pred, _ = tf.nn.seq2seq.embedding_rnn_decoder(decode_input,q_enc_state,self.cell,self.num_symbol,self.embedding_size,feed_previous=True)
        #q_mask_list = tf.unpack(tf.sequence_mask(self.input_net['q_mask'],self.q_max_length,dtype=tf.float32),axis=1)
        #for i in range(len(q_out)-1):
        #    self.output_net['loss'] += tf.nn.seq2seq.sequence_loss_by_example([q_out[i]],[decode_input[i+1]],[q_mask_list[i]])
        #self.output_net['loss'] = tf.reduce_sum(self.output_net['loss'])
        ##self.output_net['loss'] /= tf.reduce_sum(tf.cast(self.input_net['q_mask'],tf.float32))
        ##self.output_net['loss'] = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example(q_out[:-1],decode_input[1:],tf.unpack(tf.sequence_mask(self.input_net['q_mask'],self.q_max_length,dtype=tf.float32),axis=1)))
        #for i in range(len(self.q_pred)-1):
        #    self.output_net['test_loss'] += tf.nn.seq2seq.sequence_loss_by_example([self.q_pred[i]],[decode_input[i+1]],[tf.unpack(tf.sequence_mask(self.input_net['q_mask'],self.q_max_length,dtype=tf.float32),axis=1)[i]])
        #self.output_net['test_loss'] = tf.reduce_sum(self.output_net['test_loss'])
        #self.output_net['test_loss'] /= tf.reduce_sum(tf.cast(self.input_net['q_mask'],tf.float32))
        #for i in range(len(q_out)-1):
        #    self.output_net['loss'] += tf.nn.sparse_softmax_cross_entropy_with_logits(q_out[i],decode_input[i+1]) * tf.cast(tf.unpack(self.input_net['q_mask'],axis=1)[i],tf.float32)
        #self.output_net['loss'] = tf.reduce_sum(self.output_net['loss'])
        #self.output_net['loss'] /= tf.cast(tf.reduce_sum(self.input_net['q_mask']),tf.float32)
        #self.output_net['test_loss'] = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example(self.q_pred[:-1],decode_input[1:],tf.unpack(tf.sequence_mask(self.input_net['q_mask'],self.q_max_length,dtype=tf.float32),axis=1)))
        self.learning_rate = tf.Variable(0.5, trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.95)
        self.global_step = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.output_net['loss'],tvars),5.)
        self.opti = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars),global_step=self.global_step)#.minimize(self.output_net['loss'])
        #self.sch  = tf.train.AdamOptimizer(1e-3).minimize(self.output_net['test_loss'])
        init = tf.global_variables_initializer()#
        self.sess.run(init)
    def transform(self,inputs):#,outputs):
        return [tf.matmul(input,self.output_projection[0])+self.output_projection[1] for input in inputs]
    """
    def train(self,):
        sess.run()
        """
