# Created with Pyto
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from dataclasses import dataclass
import math


def initial_2D_condition_random_by_array(_domain,_source_dim):
    irow_center =random.randint(0, len(_domain)-1)
    icol_center =random.randint(0, len(_domain[0])-1)
    for i in range (irow_center-_source_dim-1,irow_center+_source_dim):
        if(i<0 or i>=len(_domain)): continue
        for j in range (icol_center-_source_dim-1,icol_center+_source_dim):
            if(j<0 or j>=len(_domain[i])): continue
            _domain[i][j]=100

def initial_2D_condition_random_by_size(size,_source_dim):
    domain = np.full((size,size),0.00)
    initial_2D_condition_random_by_array(domain,_source_dim)
    return domain

def diffusion_equation(_domain,_k,_t,_dx):
    domain_next = np.full((len(_domain), len(_domain[0])), 0.0)
    alpha = _k * _t*_dx
    for i in range(0, len(_domain)):
        for j in range(0, len(_domain[0])):
            if (i>0): domain_next[i][j]+=_domain[i-1][j]*alpha -_domain[i][j]*alpha
            if (i<len(_domain)-1): domain_next[i][j]+=_domain[i+1][j]*alpha-_domain[i][j]*alpha
            if (j>0): domain_next[i][j]+=_domain[i][j-1]*alpha-_domain[i][j]*alpha
            if (j<len(_domain[0])-1): domain_next[i][j]+=_domain[i][j+1]*alpha-_domain[i][j]*alpha
            domain_next[i][j] = _domain[i][j]+domain_next[i][j]
#    for i in range(0, len(_domain)-1):
#        for j in range(0, len(_domain[0])-1):
#            _domain[i][j]=domain_next[i][j]
    return domain_next

def get_nodeid(i,j,length):
    return i*length+length

def iterator(_duration,_timestepsize,_domain):
    for i in range (0, int(_duration/_timestepsize)):
        _domain = diffusion_equation(_domain, 0.01, _timestepsize, 1)
    return _domain

def result_writer(_pfile,_domain):
    for i in range(0, len(_domain)):
        swrite=''
        for j in range(0, len(_domain[0])):
            swrite+='%.1f' %_domain[i][j]
            swrite+=' '
        swrite+='\n'
        _pfile.write(swrite);       
    return 0

def result_reader(_pfile,_oresult_list):
    bIsInput = True
    size=0
    irowcount = 0
    iNresult_pair = 0
    while(True):
        sread=_pfile.readline()
        if(sread=="Input \n"):
            bIsInput= True
            oresult_pair=result_pair(size)
            _oresult_list.data.append(oresult_pair)
            irowcount=0
            #print("input")
        elif(sread=="Output \n"):
            bIsInput= False
            irowcount=0
            #print("output")
        elif(sread=="domain data size\n"):
            sread=_pfile.readline()
            size=int(sread)
            #print("size %d\n" %size)
        elif(not sread): break #end of file
        else:
            if (bIsInput==True):
                sread_list=sread.split(" ")[0:-1]
                for i in range (0,size-1):
                    _oresult_list.data[-1].input[irowcount][i]=float(sread_list[i])
                irowcount+=1
            else:
                sread_list=sread.split(" ")[0:-1]
                for i in range (0,size-1):
                    _oresult_list.data[-1].output[irowcount][i]=float(sread_list[i])
                irowcount+=1
    return _oresult_list
    


def Two_D_Array_Contour_Show(domain):
    plt.contourf(domain)
    plt.colorbar()
    plt.show()
    plt.clf()
    
def reconstruct_2D_array(array_1D):
    size= round(math.sqrt(len(array_1D)))
    domain = np.full((size, size), 0.0)
    for i in range (0,size):
        for j in range (0,size):
            domain[i][j]=float(array_1D[i*size+j])   
    return domain
        
class result_pair():
    def __init__(self,size):
        self.input=np.full((size,size),0)
        self.output=np.full((size,size),0)
    
class result_list():
    def __init__(self):
        self.data=[]
