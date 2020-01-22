import numpy as np
import os 
os.chdir('P:/PiotrcResearch/ColvilleUSFS_SC_subset/Zewei/')
from subprocess import call
import os.path

# number of points within each trainig sample
numofsamples={'20':2937}
def createsavesplitt(index,res):
    index=2002
    with open('T:/Users/zeweixu2/pointnet/bostonlidar/total/total'+str(index)+'.txt') as f:
        lines = f.readlines()
    nlines=[ff.split(' ') for ff in lines]
    nnlines=[[float(fff[0]),float(fff[1])] for fff in nlines]
    linesarray=np.array(nnlines)
    mini=linesarray.min(0)
    maxa=linesarray.max(0)
    xran=int(np.ceil((maxa[0] - mini[0])/res))
    yran=int(np.ceil((maxa[1] - mini[1])/res)) 
    splitt=np.zeros((xran*yran,4))
    count=-1
    for c in range(yran):
        for r in range(xran):
            count=count+1
            splitt[count,0]=mini[0]+res*c-5
            splitt[count,1]=mini[1]+res*r-5
            splitt[count,2]=splitt[count,0]+res+10
            splitt[count,3]=splitt[count,1]+res+10
    np.save('splitt'+str(index)+'_'+str(res)+'.npy',splitt) 
    return splitt
def readintxt(fil):
    with open(fil) as f:
        linesonew = f.readlines()
    nlinesonew=[ff[:-1] for ff in linesonew]
    nnlinesonew=[]
    for fff in nlinesonew:
        temp=[]
        tem=fff.split(' ')
        for i in range(len(tem)):
            item=float(tem[i])
            temp.extend([item])
        nnlinesonew.append(temp)
    linesarray=np.array(nnlinesonew)
    return linesarray
for index in [2002]:#[2009,2014]:   
    for res in [20]:
        lll=numofsamples[str(res)]
        splitt=createsavesplitt(index,res)
        if index==2002:
            shorttxt='xyzi01234'
        else:
            shorttxt='xyzir01234'
        count=-1    
        lis=[]
        countremain=0
        cco=np.zeros(len(splitt))
        cctrash=-1
        for i in range(len(splitt)):
            cctrash=cctrash+1
            print(i)
            call('P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\LAStools\\bin\\las2las.exe -i T:\\Users\\zeweixu2\\pointnet\\bostonlidar\\total\\'+'total'+str(index)+'.las'+ \
             ' -o P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\'+str(int(res))+'m'+str(index)+'\\'+str(i)+'.las'+' -keep_xy '+str(splitt[i,0])+' '+str(splitt[i,1])+' '+str(splitt[i,2])+' '+str(splitt[i,3]))
            call('P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\LAStools\\bin\\las2txt.exe -i P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\'+str(int(res))+'m'+str(index)+'\\'+str(i)+'.las'+ \
             ' -o P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\'+str(int(res))+'m'+str(index)+'\\'+str(i)+'.txt '+' -parse '+shorttxt+' -stdout')
            os.system('del P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\'+str(int(res))+'m'+str(index)+'\\'+str(i)+'.las')
            fil='P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\'+str(int(res))+'m'+str(index)+'\\'+str(i)+'.txt'            
            if os.path.exists(fil):
                array=readintxt(fil)
                if len(array)<10:
                    cco[cctrash]=1
                    countremain=countremain+1
                    if countremain==1:
                        remainarray=array
                    else:
                        remainarray=np.concatenate((remainarray,array),0)
                    continue
                elif len(array)>lll:
                    count=count+1
                    array=np.random.shuffle(array)
                    newarray=array[:lll,:]
#                    newarray=array[random.sample(xrange(0,len(array)-1), 4096),:]  
#                    rrarray=array
                    countremain=countremain+1
                    if countremain==1:
                        remainarray=array[lll:,:]
                    else:
                        remainarray=np.concatenate((remainarray,array[lll:,:]),0)                                                         
                elif len(array)<lll:
                    count=count+1
                    left=lll-len(array)
                    num=left/len(array)
                    re=left%len(array)
                    newarray=np.repeat(array,num,axis=0)
                    newarray=np.concatenate((array,newarray,array[:re,:]),axis=0)
                elif len(array)==lll:
                    count=count+1
                    newarray=array               
                if count==0:
                    totalarray=newarray.reshape((1,newarray.shape[0],newarray.shape[1]))
                else:
                    totalarray=np.concatenate((totalarray,newarray.reshape((1,newarray.shape[0],newarray.shape[1]))),axis=0)
            else:
                cco[cctrash]=1
        np.save('P:\\PiotrcResearch\\ColvilleUSFS_SC_subset\\Zewei\\totalarray'+str(res)+'.npy',totalarray)

