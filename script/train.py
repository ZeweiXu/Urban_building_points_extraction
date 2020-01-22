import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *
# croppingsize, point number in each patch and number of samples need to be added:
#20 18159 17
#30 8482 30 
#40 4916 12
#50 3241 23
#60 2289 15

##################### input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--NUM_CLASSES', type=int, default=2, help='types')
parser.add_argument('--croppingsize', type=int, default=20, help='cropping size')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=800, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
#parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()
NUM_CLASSES=FLAGS.NUM_CLASSES
croppingsize=FLAGS.croppingsize
#removetail=FLAGS.removetail
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# numofsamples: the point number threshold for different cropping sizes
numofsamples={'20':2937,'30':4782,'40':6008,'50':9516,'60':11532}
NUM_POINT=numofsamples[str(croppingsize)]
# remaining patches that need to be compensated
if croppingsize==20:
    compen=6
elif croppingsize==30:
    compen=3
elif croppingsize==40:
    compen=23
elif croppingsize==50:
    compen=19
elif croppingsize==60:
    compen=19
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
MAX_NUM_POINT = NUM_POINT
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
HOSTNAME = socket.gethostname()
from copy import copy
# calculate sample weights
def makeweights(DA):
    tmp,_ = np.histogram(DA,range(NUM_CLASSES+1))
    tmp=tmp.astype(np.float32)
    tmp=tmp/np.sum(tmp)
    tmp=1/np.log(1.2+tmp)
    ttttt=copy(DA)
    for i in range(len(tmp)):
        ttttt[ttttt==i]=tmp[i]
    return([ttttt,tmp])
# Normalize training data
def nortrain(Total,nnn):
    trainl=copy(Total[:,:,nnn]).astype(int)
    traindata=copy(Total[:,:,:nnn])
    for i in range(traindata.shape[0]):
        traindata[i]=traindata[i]-traindata[i].min(0)
    TRAIN=traindata
    return([TRAIN,trainl])

# Prepare training data
nnn=9
for lo in range(1,5):
    Total=np.load('/home/cc/pointnet2-master/sem_seg/'+str(croppingsize)+'m/'+'refdata2class_'+str(lo)+'.npy')
    [TRAIN_DATA,trl]=nortrain(Total,Total.shape[-1]-1)
    if lo==1:
        TRAIN_DATASET=TRAIN_DATA
        trainl=trl
    else:
        TRAIN_DATASET=np.concatenate((TRAIN_DATASET,TRAIN_DATA),axis=0)
        trainl=np.concatenate((trainl,trl),axis=0)   

# Prepare total patch data
TRAIN_DATASET=np.delete(TRAIN_DATASET,4,axis=2)
TEST_DATASET=TRAIN_DATASET
testl=trainl
TW=np.load('/home/cc/pointnet2-master/sem_seg/totalarray'+str(croppingsize)+'.npy')
TEST_DATASET_WHOLE=copy(TW[:,:,:nnn])
for i in range(TEST_DATASET_WHOLE.shape[0]):
    TEST_DATASET_WHOLE[i]=TEST_DATASET_WHOLE[i]-TEST_DATASET_WHOLE[i].min(0)
TEST_DATASET_WHOLE=np.concatenate((TEST_DATASET_WHOLE,TEST_DATASET_WHOLE[-compen:,:,:]),axis=0)

# Write out log files
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    #import pdb
    #pdb.set_trace()
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        # training iteration in epoches
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            # Save the variables to disk.
            if epoch % 10 == 0:
                eval_whole_scene_one_epoch(sess, ops, test_writer)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model"+str(croppingsize)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)


# Conduct one epoch of training
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(TRAIN_DATASET[:,0:NUM_POINT,:], trainl) 
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
     
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    log_string('Train_mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('Train_accuracy: %f' % (total_correct / float(total_seen)))

# evaluate one epoch of training result        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    log_string('----')
    current_data = TEST_DATASET[:,0:NUM_POINT,:]
    current_label = np.squeeze(testl)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

# evaluate on the total dataset         
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE))
    num_batches = len(TEST_DATASET_WHOLE)//BATCH_SIZE
    is_continue_batch = False
    current_data=TEST_DATASET_WHOLE
    current_label=[f*np.zeros(NUM_POINT) for f in range(len(current_data))]
    count=-1
    for batch_idx in range(num_batches):
        count=count+1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)

        if count==0:
            totallabel=pred_val
        else:
            totallabel=np.concatenate((totallabel,pred_val),axis=0)
    np.save('/home/cc/pointnet2-master/sem_seg/totallabelsquare'+str(croppingsize)+'m.npy',totallabel[:-compen,:])

# main function
if __name__ == "__main__":
    train()
    LOG_FOUT.close()
