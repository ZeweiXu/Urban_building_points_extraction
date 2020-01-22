import os
import numpy as np
os.chdir('/data/keeling/a/zeweixu2/common/zeweixu2/urbanchange/')
ccindex='total203040'
fftotal=np.load(ccindex+'.npy')

# .npy to .txt file
np.savetxt('classified'+ccindex+'.txt',fftotal,delimiter=' ',fmt='%1.4f')
from subprocess import call

# LiDAR attributes x, y, z, intensity, eigen values (normal direction in x, y ,z), curvature, and cv. 
shorttxt='xyzi01234'
call('/data/keeling/a/zeweixu2/common/sw/LAStools-master/bin/txt2las -i /data/keeling/a/zeweixu2/common/zeweixu2/urbanchange/classified'+ccindex+'.txt'+' -parse '+shorttxt+'c -add_attribute 6 "n1" "cc1" 0.000001 '+'-add_attribute 6 "n2" "cc2" 0.000001 '+'-add_attribute 6 "n3" "cc3" 0.000001 '+'-add_attribute 6 "n4" "cc4" 0.000001 '+'-add_attribute 5 "n5" "cc5" 0.000001 '+' -o /data/keeling/a/zeweixu2/common/zeweixu2/urbanchange/classified'+ccindex+'.las')


#### alternative paramters from .txt to .las file
#call('\\data\\keeling\\a\\zeweixu2\\common\\sw\\LAStools-master\\bin\\txt2las -i \\data\\keeling\\a\\zeweixu2\\common\\zeweixu2\\urbanchange\\classified'+ccindex+'.txt' \
# +' -parse '+shorttxt+'c -add_attribute 6 "n1" "cc1" 0.000001 '+'-add_attribute 6 "n2" "cc2" 0.000001 '+'-add_attribute 6 "n3" "cc3" 0.000001 '+'-add_attribute 6 "n4" "cc4" 0.000001 '+'-add_attribute 5 "n5" "cc5" 0.000001 '+ \
# ' -o \\data\\keeling\\a\\zeweixu2\\common\\zeweixu2\\urbanchange\\classified'+ccindex+'.las')

