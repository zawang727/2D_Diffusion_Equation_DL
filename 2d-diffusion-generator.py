import importlib
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from dataclasses import dataclass
modulename='2d_diffusion_tool'

iomodule=importlib.import_module(modulename)

#iterator(100,2,domain)
size = 10
pfile=open("simple_data.txt","w")
pfile.write("domain data size\n")
pfile.write("%d\n" %size)
oresult_list=iomodule.result_list()
for i in range (0,200):
    domain=iomodule.initial_2D_condition_random_by_size(size,1)
    pfile.write("Input \n")
    iomodule.result_writer(pfile,domain)
    #plt.contourf(domain)
    #plt.colorbar()
    #plt.show()
    iomodule.Two_D_Array_Contour_Show(domain)
    domain=iomodule.iterator(500,2,domain)
    pfile.write("Output \n")
    iomodule.result_writer(pfile,domain)
    iomodule.Two_D_Array_Contour_Show(domain)
    print("i %d" %i)
    #plt.contourf(domain)
    #plt.colorbar()
    #plt.show()
pfile.close()
pfile=open("data.txt","r")
oresult=iomodule.result_list()
oresult=iomodule.result_reader(pfile,oresult)
pfile.close()
#print(oresult.data[0].input)
#Two_D_Array_Contour_Show(oresult.data[3].input)
#Two_D_Array_Contour_Show(oresult.data[3].output)
