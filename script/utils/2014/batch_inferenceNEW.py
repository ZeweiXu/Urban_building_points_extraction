import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *

# python batch_inferenceNEW.py --model_path log/model.ckpt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=10000, help='Point number [default: 10000]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--data',default='/home/cc/pointnet-master/sem_seg/points687.npy',help='testing data')
FLAGS = parser.parse_args()
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
testdatapath=FLAGS.data

NUM_CLASSES = 9

#def log_string(out_str):
#    LOG_FOUT.write(out_str+'\n')
#    LOG_FOUT.flush()
#    print(out_str)

def evaluate():
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
 #   log_string("Model restored.")
    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    #import pdb
    #pdb.set_trace()

    prediction_label = predd(sess, ops)
    outname=testdatapath.split('/')[-1].split('.')[0][1:5]
    np.save('/home/cc/pointnet-master/sem_seg/predicted_label_'+outname+'.npy',prediction_label)    

def predd(sess, ops):
    is_training = False
    current_data =np.load(testdatapath)[:,:,:3]
    current_label=np.ones((current_data.shape[0],current_data.shape[1]))
    for batch_idx in range(current_data.shape[0]):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx].astype(int),
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        if batch_idx==0:
            pred_label = np.argmax(pred_val, 2) 
        else:
            pred_label=np.concatenate((pred_label,np.argmax(pred_val, 2)),axis=0)
    
    return pred_label


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
#    LOG_FOUT.close()
