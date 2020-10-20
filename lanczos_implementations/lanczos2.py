import numpy as np #ALL LINEAR ALGEBRA
import scipy as sc #FOR SPARSE MATRIX
import numba #PARALELISM, CUNCURRENCE AND GPU
import time as t

#CODE FOR LANCZOS TRANSFORMATION FOR SQUARE AND HEXAGONAL LATTICES
#THE FILE PLOTS NEEDS OF ALL THE RETURNS OF LANCOZS FUNCTIONS (IF TRUE, WILL BE NECESSARY MATPLOTLIB, PANDAS AND SEABORN)


def modu(vec): #SIMPLE FUNC TO CALCULATE VEC^2
    return(np.inner(vec,vec)) #vec.vec -> x^2 + y^2 + z^2


def bracket(a,b,c): #A^{\dagger} . b .A
    return(np.inner(np.transpose(np.conj(a)),np.dot(b,c))) #EXPECTED VALUE FROM QUANTUM MECHANICS


def lanczos_sys_sol(A,m=0): #SOLVES THE LINEAR PROBLEM FROM MATRIX MANIPULATION
        #it's the same method on https://www.bing.com/search?q=Kondo+versus+indirect+exchange%3A+Role+of+lattice+and+actual+range+of+RKKY+interactions+in+real+materials&FORM=ANCMS9&PC=U531
    if m==0:
        m=len(A)
    
    lb=int(np.sqrt(m)**2-4*np.sqrt(m)+4) #lanczos base dim
    interactions=int((np.sqrt(m)-1)/2) #number of interactions 
     
    
    #columns vectors
    alpha=np.zeros((m,interactions),dtype=np.complex64) #alphas vectors
    beta=np.zeros((m,interactions),dtype=np.complex64) #betas...

    #where is the seed?
    sed1 = 25 #first one
    sed2 = 35 #second seed
    
    aux1 = np.zeros(m)
    aux1[sed1] = 1 
    
    aux2 = np.zeros(m)
    aux2[sed2] = 1
    
    alpha[:,0] = aux1 #firt seed 
    beta[:,0] = aux2 #second seed 
    

    a=np.zeros((interactions,2,2),dtype=np.complex64) # "A" MATRICES FROM BLOCK-DIAGONAL PART 
    b=np.zeros((interactions,2,2),dtype=np.complex64) # "B" MATRICES FROM NON-BLOCK-DIAGONAL PART



    #STEPS ZERO AND ONE, BECAUSE LANCZOS METHODS NEEDS "N" AND "N-1" TO DISCOVER "N+1"
    a[0,:,:] = np.array(([bracket(alpha[:,0],A,alpha[:,0]),bracket(beta[:,0],A,alpha[:,0])],[bracket(alpha[:,0],A,beta[:,0]),bracket(beta[:,0],A,beta[:,0])]))@np.linalg.inv(np.array(([modu(alpha[:,0]),np.inner(beta[:,0],alpha[:,0])],[np.inner(beta[:,0],alpha[:,0]),modu(beta[:,0])])))
    
    alpha[:,1] = A@alpha[:,0] - a[0,0,0]*alpha[:,0] - a[0,1,0]*beta[:,0]
    beta[:,1] = A@beta[:,0] - a[0,1,1]*beta[:,0] - a[0,0,1]*alpha[:,0]

    
    a[1,:,:] = np.array(([bracket(alpha[:,1],A,alpha[:,1]),bracket(beta[:,1],A,alpha[:,1])],[bracket(alpha[:,1],A,beta[:,1]),bracket(beta[:,1],A,beta[:,1])]))@np.linalg.inv(np.array(([modu(alpha[:,1]),np.inner(beta[:,1],alpha[:,1])],[np.inner(beta[:,1],alpha[:,1]),modu(beta[:,1])])))
    b[1,:,:] = np.array(([bracket(alpha[:,1-1], A, alpha[:,1]),bracket(beta[:,1-1],A,alpha[:,1])],[bracket(alpha[:,1-1],A,beta[:,1]),bracket(beta[:,1-1], A, beta[:,1])]))@np.linalg.inv(np.array(([modu(alpha[:,1-1]),np.inner(beta[:,1-1],alpha[:,1-1])],[np.inner(alpha[:,1-1],beta[:,1-1]),
                                                                                                                                                                                                                                                            modu(beta[:,1-1])])))
    


    alpha[:,2] = A@alpha[:,1] - a[1,0,0]*alpha[:,1] - a[1,1,0]*beta[:,1] - b[1,0,0]*alpha[:,0] - b[1,0,1]*beta[:,0]
    beta[:,2] = A@beta[:,1] - a[1,1,1]*beta[:,1] - a[1,0,1]*alpha[:,1]- b[1,1,1]*beta[:,0] - b[1,1,0]*alpha[:,0]
    
    
    
    for i in range(2,interactions-1):
        
        a[i,:,:] = np.array(([bracket(alpha[:,i],A,alpha[:,i]),bracket(beta[:,i],A,alpha[:,i])],[bracket(alpha[:,i],A,beta[:,i]),bracket(beta[:,i],A,beta[:,i])]))@np.linalg.inv(np.array(([modu(alpha[:,i]),np.inner(beta[:,i],alpha[:,i])],[np.inner(beta[:,i],alpha[:,i]),modu(beta[:,i])])))
        b[i,:,:] = np.array(([bracket(alpha[:,i-1], A, alpha[:,i]),bracket(beta[:,i-1],A,alpha[:,i])],[bracket(alpha[:,i-1],A,beta[:,i]),bracket(beta[:,i-1], A, beta[:,i])]))@np.linalg.inv(np.array(([modu(alpha[:,i-1]),np.inner(beta[:,i-1],alpha[:,i-1])],[np.inner(alpha[:,i-1],beta[:,i-1]),modu(beta[:,i-1])])))
    
    
        alpha[:,i+1] = A@alpha[:,i] - a[i,0,0]*alpha[:,i] - a[i,1,0]*beta[:,i] - b[i,0,0]*alpha[:,i-1] - b[i,0,1]*beta[:,i-1]
        beta[:,i+1] = A@beta[:,i] - a[i,1,1]*beta[:,i] - a[i,0,1]*alpha[:,i]- b[i,1,1]*beta[:,i-1] - b[i,1,0]*alpha[:,i-1]
        
    for i in range(interactions):
        alpha[:,i] = alpha[:,i]/np.sqrt(modu(alpha[:,i]))
        beta[:,i] = beta[:,i]/np.sqrt(modu(beta[:,i]))
        
        beta[:,i] = beta[:,i] - (np.inner(beta[:,i],alpha[:,i])/modu(alpha[:,i]))*alpha[:,i]
    
    
    data=np.zeros((interactions,4),dtype=np.complex)
    data[1:,0] = np.diag(np.transpose(np.conj(alpha))@A@alpha,k=1)
    data[1:,1] = np.diag(np.transpose(np.conj(alpha))@A@alpha,k=-1)
    data[:,2] = np.diag(np.transpose(np.conj(alpha))@A@beta,k=0)
    data[:,3] = -1*np.diag(np.transpose(np.conj(beta))@A@alpha,k=0)
    np.savetxt("output/"+str(sed1)+"_"+str(sed2)+"__data_real.dat",np.real(data),fmt='%+.14f',header = "b++             b+-               a++               a+-")
    np.savetxt("output/"+str(sed1)+"_"+str(sed2)+"__data_imag.dat",np.imag(data),fmt='%+.14f',header = "b++             b+-               a++               a+-")

    return(alpha,beta,a,b)


def lanczos_rkky_norht(A,m=0): #LANCZOS OF THE BELLOW PAPER, WITHOUT RE-ORTHOGONALIZATION 
#ONE SEED
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
    alpha[0]=0. #np.dot(np.dot(np.conj(psi[:,0]),A),psi[:,0])/modu(psi[:,0]) #alpha_0
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
    
    num_elem=int(matriz.max())
    matriz=np.array(matriz,dtype=np.int64)
    shape_real=np.shape(matriz)[0]
    #print(vec)
    
    
    
    hamil=np.zeros((num_elem+1,num_elem+1),dtype=np.complex)
    
    matriz=np.c_[np.zeros(len(matriz)),matriz,np.zeros(len(matriz))]
    matriz=np.r_[[np.zeros(len(matriz[0,:]))],matriz,[np.zeros(len(matriz[0,:]))]]
    
    matriz=np.c_[np.zeros(len(matriz)),matriz,np.zeros(len(matriz))]
    matriz=np.r_[[np.zeros(len(matriz[0,:]))],matriz,[np.zeros(len(matriz[0,:]))]]
        
    
    locs = np.zeros((int(num_elem/2)+1,3))
    vec=np.transpose(np.nonzero(matriz))
    t=1.0 #hopping
    st =0.1j #second hopping
    #return matrix
    for i in range(int(num_elem/2)+1):
        lin=vec[i][0]
        col=vec[i][1]
        val=int(matriz[lin,col])
        locs[i,0] = val
        locs[i,1] = lin
        locs[i,2] = col
                
        hopps=matriz[lin-1:lin+2,col-1:col+2]
        aft=int(hopps[1,2])
        dia_sup=int(hopps[0,2])
        dia_inf=int(hopps[2,2])
        
        hamil[val,aft]=t
        hamil[val,dia_inf]=t
        hamil[val,dia_sup]=t
    
    for i in range(int(num_elem/2)+1):
        
        lin=vec[i][0]
        col=vec[i][1]
        val=int(matriz[lin,col])        

        shopps=matriz[lin-2:lin+3,col-2:col+3]
        
        Aabove_above=int(shopps[0,2])
        Aabove_rigt=int(shopps[1,-1])
        Abelow_right=int(shopps[3,-1])
        
        Babove_above=int(shopps[0,1])
        Babove_rigt=int(shopps[1,3])
        Bbelow_right=int(shopps[3,3])
        
        hamil[val,Aabove_above]=st
        hamil[val,Aabove_rigt]=st
        hamil[val,Abelow_right]=-st
        
        
        #hamil[val,Babove_above]=st
        #hamil[val,Babove_rigt]=st
        #hamil[val,Bbelow_right]=-st

        
        
    #hamil=hamil[1:-1,1:-1]
    np.savetxt("localizacao.txt",locs)
    hamil=hamil+np.transpose(np.conj(hamil))
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
    
    
    hopp=hopp[:-1,:-1]
    hopp=hopp[0:,0:]
    
    hopp=hopp+np.transpose(np.conj(hopp))
    return hopp

spi=spiral_honney(21,21) # GENERATE THE LATTICE
ham=hamiltonian_honney(spi) #TRANSFORM INTO A HAMILTONIAN
lanczos_sys_sol(ham)
