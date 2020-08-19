import numpy as np
import time as t
np.set_printoptions(precision=32,suppress=False) #float precision to print
import scipy as sc
import numba


def modu(vec):
    return(np.inner(vec,vec))

def lanczos_rkky_norht(A,m=0): #a few diferent implementation of the method with any orthogonalization
    #I'm fallowed the paper  https://doi.org/10.3389/fphy.2019.00067  
    if m==0:
        m=len(A)

    lb=int(np.sqrt(m)**2-4*np.sqrt(m)+4) #lanczos base dim
    interactions=int((np.sqrt(m)-1)/2) #number of interactions 
    #define your seed
    psi=np.zeros((m,interactions)) #memory allocation for psi's 
    alpha=np.zeros(interactions) #memory allocation for beta and alpha vectors
    beta=np.zeros(interactions) 
    seed=np.zeros(m) #seed vector
    
    seed_site=0 #seed where the impurity is
    seed[seed_site]=1. #whats the norm of the seed
    psi[:,0]=seed #seed
    
    #first step
    alpha[0]=np.dot(np.dot(np.conj(psi[:,0]),A),psi[:,0])/modu(psi[:,0]) #alpha_0
    psi[:,1]=np.dot(A,psi[:,0])-alpha[0]*psi[:,0] #psi_{n+1}
    beta[0]=.0
    beta[1]=(np.linalg.norm(psi[:,1])**2)/modu(psi[:,0]) #beta_1
    #next steps
    for i in range(2,interactions):
              
        psi[:,i]=np.dot(A,psi[:,i-1])-alpha[i-1]*psi[:,i-1]-(beta[i-1])*psi[:,i-2]
        
        
        beta[i]=modu(psi[:,i])/modu(psi[:,i-1])
        alpha[i]=np.dot(np.dot(np.transpose(psi[:,i]),A),psi[:,i])/modu(psi[:,i])
     
        for j in range(i): #gram-schmidt orthogonalization
            psi[:,i]=psi[:,i]-(np.inner(psi[:,i],psi[:,j])/modu(psi[:,j]))*psi[:,j]
        


    for i in range(interactions):
        psi[:,i]=psi[:,i]/np.linalg.norm(psi[:,i])



    beta=np.sqrt(beta) #that's a problem of np.eye*b
    aux=np.zeros(len(beta)) #I'm just fixing that with this auxiliar vector
    aux[:-1]=beta[1:]
    beta=aux
    
    
    B=np.eye(len(alpha))*alpha+np.eye(len(beta),k=-1)*beta[:]+np.transpose(np.eye(len(beta),k=-1)*beta[:])
    return(B,psi,A)

def test_matrix(dim): #that's useful to test the code, it generates a hermitian matrix
    alea=np.random.rand(dim,dim)
    return (0.5*(alea+np.transpose(alea)))

def spiral(width, height): #it generates a espiral clockwise matrix
    NORTH, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0) # directions
    turn_right = {NORTH: E, E: S, S: W, W: NORTH} # old -> new direction
                                                  #need's 
    if width !=height:
        raise ValueError

    x, y = width // 2, height // 2 # start near the center
    dx, dy = NORTH # initial direction
    matrix = [[None] * width for _ in range(height)]
    count = 0
    while True:
        count += 1
        matrix[y][x] = count # visit
        # try to turn right
        new_dx, new_dy = turn_right[dx,dy]
        new_x, new_y = x + new_dx, y + new_dy
        if (0 <= new_x < width and 0 <= new_y < height and
            matrix[new_y][new_x] is None): # can turn right
            x, y = new_x, new_y
            dx, dy = new_dx, new_dy
        else: # try to move straight
            x, y = x + dx, y + dy
            if not (0 <= x < width and 0 <= y < height):
                return (np.array(matrix)[0:,0:]) # nowhere to go


def honney(x,y): #y must to be something like 2*n+1, where 'n' is the number of unity cells
    
    l=[[],[]]
    l[0].append([0,1,1,0]*x) #rows
    l[1].append([1,0,0,1]*x)

    
    phy_bas=np.zeros((y,len(l[0][0])))
    
    #print((l[0][0]))
    #print((l[1][0]))
    
    phy_bas[0,:]=l[0][0]
    for i in range(1,y,2):
        phy_bas[i,:]=l[1][0]
        phy_bas[i+1,:]=l[0][0]
        
    print(phy_bas)
    mem=1 #auxiliar number no index the lattice
    for i in range(y):
        for j in range(len(l[1][0])):
            if phy_bas[i,j]!=0:
                phy_bas[i,j]=mem
                mem+=1
    return phy_bas,mem #return the matrix and the number of elements in the system



def hamiltonian_honney(matriz,num_ele): #matrix and number of elements
    hamil=np.zeros((num_ele,num_ele))
    

def hamiltonian_square(matriz): #transform the abose matrix into the hamiltonian with dim=width*height
    base=len(matriz)**2
    hopp=np.zeros((base+1,base+1))
    t=1.0 #hopp energy
    
    matriz=np.c_[np.zeros(len(matriz)),matriz,np.zeros(len(matriz))]
    
    matriz=np.r_[[np.zeros(len(matriz[0,:]))],matriz,[np.zeros(len(matriz[0,:]))]]
    
    
    for i in range(len(matriz[:,0])):
        for j in range(len(matriz[0,:])):
            if matriz[i,j]!=0:
                val=int(matriz[i,j])
                
                infe=int(matriz[i+1,j])
                sup=int(matriz[i-1,j])
                left=int(matriz[i,j-1])
                right=int(matriz[i,j+1])
                
                if infe!=0:
                    hopp[infe-1,val]=t
                if left!=0:
                    hopp[val-1,left]=t
                if right!=0:
                    hopp[val-1,right]=t
                if sup!=0:
                    hopp[sup-1,val]=t
                #print(hopp)
    
    for i in range(len(hopp[:,0])):
        hopp[i,i+1:]=0
    
    
    hopp=hopp[:-1,1:]
    hopp=hopp+np.transpose(hopp)
    return hopp



a=honney(3,3)











































'''
t0=t.time()
var=11
spi=spiral(var,var)
h=hamiltonian2(spi)
print(t.time()-t0)

t0=t.time()
a,b,c=lanczos_rkky_norht(h)
print(t.time()-t0)


import seaborn as sea
import pandas as pd
import matplotlib.pyplot as plt

#q=np.dot(np.transpose(b),b)


plt.clf()

plt.subplot(221)
plt.title(r"Lanczos Basis $\psi's$")
sea.heatmap(pd.DataFrame(b),cmap='hot',annot=False)

plt.subplot(211)
plt.title(r"Coefficients directly from Lanczos Algorithm")

sea.heatmap(pd.DataFrame(a),cmap='hot',annot=False)

plt.subplot(223)
plt.title(r"$\psi^{\dagger }H\psi$")
sea.heatmap(np.dot(np.transpose(b),np.dot(c,b)),cmap='hot',annot=False)


plt.subplot(224)
plt.title(r"$\psi^{\dagger }\psi$")
sea.heatmap(np.dot(np.transpose(b),b),cmap='coolwarm',annot=False)


plt.tight_layout()
plt.savefig('results.pdf')
'''