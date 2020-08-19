#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <bits/stdc++.h> 


using namespace std; 


int *spiral (int n)
{
    int *matrix=new int [n];
    int count=0;
    //int *matrix=(int*)malloc(n*sizeof(int));
    
    for (int i = 0; i < n; i++) 
    { 
        for (int j = 0; j < n; j++) 
        { 
            int x; 
  
            x = min(min(i, j), min(n-1-i, n-1-j)); 
  
            if (i <= j) 
                matrix[j+count]=(n-2*x)*(n-2*x) - (i-x) - (j-x); 
  
            else
                matrix[j+count]=(n-2*x-2)*(n-2*x-2) + (i-x) + (j-x); 
        }


        count+=n;
    }


    return (matrix);

}
