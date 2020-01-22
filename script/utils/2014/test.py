import os
import numpy as np
os.chdir('/home/cc/pointnet2-master/sem_seg/')
#for i in range(10):
#    temp=np.load('totalarray'+str(i)+'.npy')
#    if i==0:
#        total=temp
#    else:
#        total=np.concatenate((total,temp),axis=0)
#np.save('total2002.npy',total)
#import pdb
#pdb.set_trace()
data=np.load('/home/cc/pointnet2-master/sem_seg/total2002.npy')[:-8]
label=np.load('/home/cc/pointnet2-master/sem_seg/totallabelsquare.npy')
for i in range(len(data)):
    if i==0:
        tem=np.unique(np.concatenate((data[i][:,:3],label[i].reshape(len(label[i]),1)),axis=1),axis=0)
        totalarray=tem
    else:
        tem=np.unique(np.concatenate((data[i][:,:3],label[i].reshape(len(label[i]),1)),axis=1),axis=0)
        totalarray=np.concatenate((totalarray,tem),axis=0)
np.savetxt('finalarray2002.txt',totalarray,delimiter=' ',fmt='%1.4f')

