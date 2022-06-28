import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from cmcrameri import cm

folder="..."
file_name="SIM_lambda540nm_XY"
file_path=folder+file_name+".txt"
pixels=401

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

rc('font', **font)

data=np.loadtxt(file_path,dtype=str,delimiter="\n")

x=np.array(data[2:pixels+2],dtype=float)
y=np.array(data[pixels+3:2*pixels+3],dtype=float)

E=np.zeros((0,pixels))
temp=np.array(data[2*pixels+4:][0:])
for line in temp:
    add_line=np.fromstring(line,dtype=float,sep=" ")
    E=np.append(E,np.expand_dims(add_line,0),axis=0)

E=E.transpose()

plt.figure()
plt.imshow(E, interpolation='antialiased',cmap=cm.batlow,extent=[x[0]*1e9,x[-1]*1e9,y[0]*1e9,y[-1]*1e9],norm=colors.LogNorm())
cbar=plt.colorbar()
cbar.set_label(label='Field enhancement [ |E|\u00b2 / |E\u2070|\u00b2 ]',rotation=270, labelpad=20)
plt.axis('off')
plt.savefig(folder+"/plots/"+file_name+'map.png',dpi=300)
plt.show()
