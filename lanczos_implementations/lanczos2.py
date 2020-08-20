import numpy as np #ALL LINEAR ALGEBRA
import scipy as sc #FOR SPARSE MATRIX
import numba #PARALELISM, CUNCURRENCE AND GPU
import time as t

#CODE FOR LANCZOS TRANSFORMATION FOR SQUARE AND HEXAGONAL LATTICES
#THE FILE PLOTS NEEDS OF ALL THE RETURNS OF LANCOZS FUNCTIONS (IF TRUE, WILL BE NECESSARY MATPLOTLIB, PANDAS AND SEABORN)


def modu(vec): #SIMPLE FUNC TO CALCULATE VEC^2
    return(np.inner(vec,vec))

def lanczos_rkky_norht(A,m=0): #LANCZOS OF THE BELLOW PAPER, WITHOUT RE-ORTHOGONALIZATION
    #I'm fallowed the paper  https://doi.org/10.3389/fphy.2019.00067  
    #Here our hamiltoninan don't have orbital energy, so alpha_n=0
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
    alpha[0]=0 #np.dot(np.dot(np.conj(psi[:,0]),A),psi[:,0])/modu(psi[:,0]) #alpha_0
    psi[:,1]=np.dot(A,psi[:,0])-alpha[0]*psi[:,0] #psi_{n+1}
    beta[0]=.0
    beta[1]=(np.linalg.norm(psi[:,1])**2)/modu(psi[:,0]) #beta_1
    #next steps
    for i in range(2,interactions):
              
        psi[:,i]=np.dot(A,psi[:,i-1])-alpha[i-1]*psi[:,i-1]-(beta[i-1])*psi[:,i-2]
        
        
        beta[i]=modu(psi[:,i])/modu(psi[:,i-1])
        alpha[i]=0 #np.dot(np.dot(np.transpose(psi[:,i]),A),psi[:,i])/modu(psi[:,i])
     
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

def spiral_honney(width, height): #GENERATE A SPIRAL LATTICE WITH HONEYCOMB CONFIGURATION, IN CLOCKWISE DIRECTION
                                  #WIDTH AND HEIGHT MUST BE EQUAL AND,
                                  # WIDTH NEED TO RESPECT THE IDEIA OF ((WIDTH-1)/4)%4)==0
                                  
    NORTH, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0) # directions
    turn_right = {NORTH: E, E: S, S: W, W: NORTH} # old -> new direction
                                                  #need's 
    if width !=height:
        raise ValueError

    x, y = width // 2, height // 2 # start near the center
    dx, dy = NORTH # initial direction
    matrix = [[None] * width for _ in range(height)]
    count = 0
    antes =0 
    while True:
        count+=1
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
                (np.array(matrix)[0:,0:]) # nowhere to go
                break
            
    matrix=np.array(matrix)
    mul=int((width/4))
    l1=[1,0,0,1]*mul
    l2=[0,1,1,0]*mul
    
    l1.append(1)
    l2.append(0)
    for i in range(0,height,2):
        matrix[i,:]=matrix[i,:]*l1
    
    for i in range(1,height,2):
        matrix[i,:]=matrix[i,:]*l2
    return matrix
def spiral(width, height): #GENERATE A WIDTH X WIDTH (WIDTH%2==1) LATTICE WITH CLOCKWISE ORIENTATION
                            # 
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
    
    
    
    
def hamiltonian_honney(matriz): #TRANSFORM THE HONNEYCOMB LATTICE INTO A HAMILTONIAN (FIRST HOPPING ONLY)
    
    #matriz=np.round(0.5+matriz/2,0) #I'm excluding the non-zeros points of the count
    vec=np.transpose(np.nonzero(matriz))
    num_elem=int(matriz.max())
    matriz=np.array(matriz,dtype=np.int64)
    shape_real=np.shape(matriz)[0]
    #print(vec)
    
    
    
    hamil=np.zeros((num_elem+2,num_elem+2))
    
    matriz=np.c_[np.zeros(len(matriz)),matriz,np.zeros(len(matriz))]
    
    matriz=np.r_[[np.zeros(len(matriz[0,:]))],matriz,[np.zeros(len(matriz[0,:]))]]
    
    
    vec=np.transpose(np.nonzero(matriz))
    
    t=1.0 #hopping
    #return matriz
    

    for i in range(int(num_elem/2)):
        lin=vec[i][0]
        col=vec[i][1]
        val=int(matriz[lin,col])
        
        hopps=matriz[lin-1:lin+2,col-1:col+2]
        aft=int(hopps[1,2])
        dia_sup=int(hopps[0,2])
        dia_inf=int(hopps[2,2])
        
        hamil[val,aft]=t
        hamil[val,dia_inf]=t
        hamil[val,dia_sup]=t
    
    hamil=hamil[1:-1,1:-1]
    hamil=hamil+np.transpose(hamil)
    return hamil
        
def hamiltonian_square(matriz): #TRANSFORM THE SQUARE LATTICE INTO A HAMILTONIAN (FIRST HOPPING ONLY)
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

t0=t.time()
spi=spiral_honney(81,81) # GENERATE THE LATTICE
ham=hamiltonian_honney(spi) #TRANSFORM INTO A HAMILTONIAN
a,b,c=lanczos_rkky_norht(ham) #MAKES LANCZOS TRANSFORMATION 
PLOTS=False #DO YOU TO PLOT?
SAVE_DATA=True
print(t.time()-t0)
if SAVE_DATA:
    np.savetxt('lanczos_coeffs.dat',a)
    
    np.savetxt('lanczos_psis.dat',b)
    
    np.savetxt('hamiltonian_utilized.dat',c)



if PLOTS:
    import seaborn as sea
    import pandas as pd
    import matplotlib.pyplot as plt
    
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
    

