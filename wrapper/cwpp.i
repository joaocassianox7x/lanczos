
/* name of module to use*/
%module cwpp
%{ 
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "cwpp.h" 
  
%} 
  

int *spiral (int n);

