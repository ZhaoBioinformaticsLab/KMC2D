/* All the source */
#include <iostream> // Cin/ Cout
#include <fstream>  // For Read/Write File
#include <string>
#include <sstream>  // using sstream
#include <vector>   // For using Vector
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
using namespace std;
#define TILE_DIM 32    // For 2D Thread Block Allocation 
#define Block_Size 128 // For 1D Thread Block Allocation 
enum run_mode {GPU_PARA=0, CPU_GOLDEN, READ_THRU};

//int parse_cmd_line(int argc, char* argv[], string & G_File_Name, string & K_File_Name, int & Individual_Num, int & MarkerBlock_Size);

int parse_cmd_line(int argc, char* argv[], string & Z_File_Name, string & W_File_Name, string & K_File_Name, int & Individual_Num, int & Marker_Num_Length, int & MarkerPair_Block_Size, char & Run_Mode)
{ 
	if((argc!=2)&&(argc <15))
	{
		cerr <<"Please specific the correct parameters, or use parameter -u for usermanuals!" <<endl;
		return 1;
	}
	else
	{
	   for (int i=1; i<argc;i++)
       {
	       string arg=argv[i];
	       if((arg=="-z")||(arg=="-Z"))
	       {
             if(i+1<argc)
		     {
		       Z_File_Name =argv[i+1];
		     }
		     else
		     {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the full file name of the first genotype matrix!" <<endl;
			   return 1; // Parse cmd_line fail
		      }
		    }	
		    
		   if((arg=="-w")||(arg=="-W"))
	       {
             if(i+1<argc)
		     {
		       W_File_Name =argv[i+1];
		     }
		     else
		     {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the full file name of the second genotype matrix!" <<endl;
			   return 1; // Parse cmd_line fail
		      }
		   }	
		    
		   if((arg=="-k")||(arg=="-K"))
	       {
             if(i+1<argc)
		     {
		       K_File_Name =argv[i+1];
		     }
		     else
		     {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the full file name of the output kinship matrix !" <<endl;
			   return 1; // Parse cmd_line fail
		      }
		    }        

            if((arg=="-i")||(arg=="-I"))
	        {
              if(i+1<argc)
		      {
		           Individual_Num =atoi(argv[i+1]);
		      }
		      else
		      {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the Individual Number!" <<endl;
			   return 1; // Parse cmd_line fail
		      }
	        }
	        
	        if((arg=="-l")||(arg=="-L"))
	        {
              if(i+1<argc)
		      {
		           Marker_Num_Length =atoi(argv[i+1]);
		      }
		      else
		      {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the Direct Marker Number Length!" <<endl;
			   return 1; // Parse cmd_line fail
		      }
	        }
	        
	        if((arg=="-b")||(arg=="-B"))
	        {
              if(i+1<argc)
		      {
		         MarkerPair_Block_Size =atoi(argv[i+1]);
		      }
		      else
		      {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the Marker Pair Block Size!" <<endl;
			   return 1; // Parse cmd_line fail
		      }
	        }
	        
	        if((arg=="-m")||(arg=="-M"))
	        {
              if(i+1<argc)
		      {
		           Run_Mode =atoi(argv[i+1]);
		      }
		      else
		      {
			   cerr<<"Parse cmd_line fail, Need to clearly specific the Run Mode(0: GPU Paral; 1: Golden CPU; 2: Read thorugh) !" <<endl;
			   return 1; // Parse cmd_line fail
		      }
	        } 

		   if((arg=="-h")||(arg=="-H"))
	       {
			     cout<<"Welcome to use this program to do Kinship Matrix Calculation" <<endl;
				 cout << "The usage of input parameter arguments are listed as followings:" <<endl;
				 cout << "-h or -H: Output this Help usage message" <<endl;
				 cout << "-z or -Z: The full name of the 1st Genotype file"<<endl;
				 cout << "-w or -W: The full name of the 2nd Genotype file"<<endl;
				 cout << "-k or -K: The full name of Kinship Matrix file"<<endl;
				 cout << "-i or -I: The Individual number" <<endl;
				 cout << "-l or -L: The direct marker Length" <<endl;
				 cout << "-b or -B: The Block size for dividing the genotype marker pairs " <<endl;
				 cout << "-m or -M: Run_Mode(0, 1, 2 for the GPU/CPU/Read Through) " <<endl;
			     return 1; // Parse cmd_line fail
		    }
	    }
	}
    return 0;
}

__global__ void Genotype_Matrix_Transpose(float *GMatrix_O, float *GMatrix_I, int Matrix_Width, int Matrix_Height) //ok
{
   __shared__ float block[TILE_DIM][TILE_DIM+1];
   int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
   int Index_In = xIndex + (yIndex)*Matrix_Width;
    
   if( (xIndex < Matrix_Width) && (yIndex < Matrix_Height))
      block[threadIdx.y][threadIdx.x] = GMatrix_I[Index_In];
   else    
      block[threadIdx.y][threadIdx.x] = 0.0; //pad zero for the unarbitrary matrix 
          
   __syncthreads();
    
   xIndex = blockIdx.y * TILE_DIM + threadIdx.y;
   yIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   int Index_Out = xIndex + (yIndex)*Matrix_Height;
   if( (xIndex < Matrix_Height)&&(yIndex < Matrix_Width))
      GMatrix_O[Index_Out] = block[threadIdx.y][threadIdx.x];     
      
   __syncthreads();  
     
}

__global__ void Kinship_Matrix_Cal(float *Genotype_Matrix_Transpose_In, float *Genotype_Matrix_In, float *Kinship_Matrix_Out,int Individual_Num,  int Marker_Num) 
{
   /* Use shared memory to calculate t(GM)*GM, 
    * GM is original genotype matrix, which is a mxn matrix and m is marker number; 
    * t(GM) means the transpose of the original genotype matrix, which is a nXm matrix  */
      
   __shared__ float t_s[TILE_DIM][TILE_DIM];   // shared memory for transposed matrix
   __shared__ float m_s[TILE_DIM][TILE_DIM];   // shared memory for matrix 
   
   int r_t     = TILE_DIM* blockIdx.y+ threadIdx.y; // row index for transposed matrix 
   int c_t     = threadIdx.x;                       // column index for transposed matrix
   int r_m     = threadIdx.y;                       // row index of matrix
   int c_m     = blockIdx.x* TILE_DIM+  threadIdx.x;// column index for matrix                  
   
   float entry_sub =0.0;  
   for(int i=0; i< (Marker_Num +TILE_DIM-1)/TILE_DIM; i++)
   {
	   float value_t, value_m;
	   int index_t = r_t*Marker_Num +c_t;         //linear index of transposed matrix
	   int index_m = r_m* Individual_Num+ c_m;    //linear index of genotype matrix
	   
	   if ((r_t < Individual_Num )&&(c_t < Marker_Num))	   
	       value_t = Genotype_Matrix_Transpose_In[index_t];	   
	   else	   
		   value_t =0.0;  // pad with 0
	   
	   if ((r_m< Marker_Num)&&(c_m < Individual_Num))
	       value_m = Genotype_Matrix_In[index_m];
	   else
	       value_m =0.0;  // pad with 0
	   
	   c_t += TILE_DIM;
	   r_m += TILE_DIM;    
	   t_s[threadIdx.y][threadIdx.x]  = value_t; 
	   m_s[threadIdx.y][threadIdx.x]  = value_m;	   
	   __syncthreads();
	   
	   for (int k=0; k< TILE_DIM; k++)
	     entry_sub += t_s[threadIdx.y][k]* m_s[k][threadIdx.x];
	     
	   __syncthreads();
	   	  
   }
   
   int r_k     = TILE_DIM* blockIdx.y+ threadIdx.y; // row index for kinship matrix;
   int c_k     = TILE_DIM* blockIdx.x+ threadIdx.x; // column index for kinship matrix;
   int index_k = r_k* Individual_Num+ c_k;          //linear index of kinship matrix
   
   if((r_k<Individual_Num) &&(c_k<Individual_Num))
	   Kinship_Matrix_Out[index_k] =entry_sub;
	   
   __syncthreads();
  
}

__global__ void Kinship_Matrix_Add(float *Kinship_Matrix_Sub, float *Kinship_Matrix, long Matrix_Size) 
{  
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   long stride = blockDim.x * gridDim.x; 
   for (long i = index; i < Matrix_Size; i += stride)
   Kinship_Matrix[i] = Kinship_Matrix[i] +  Kinship_Matrix_Sub[i];
   
}

__global__ void Kinship_Matrix_Normalize(float *Kinship_Matrix, long Matrix_Size, float Normalize_Rate) 
{
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   long stride = blockDim.x * gridDim.x; 
   for (long i = index; i < Matrix_Size; i += stride)
      Kinship_Matrix[i] = Kinship_Matrix[i]/Normalize_Rate ;
}

__global__ void Matrix_AllValue_Set(float *Matrix, long Matrix_Size, float Value_Set) 
{
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x; 
   
   for (long i = index; i < Matrix_Size; i += stride)
       Matrix[i] = Value_Set;
}

/*The golden code implementation of kinship matrix calculoation, which can be used for the base evaluation of GPU paralleling*/
void Golden_Kinship_Matrix_Cal(float *Kinship_Matrix_Out, float *GMatrix_I,  int Individual_Num,  int Marker_Num)
{
	for (int i =0; i< Individual_Num; i++)
	{		
	   for (int j=0; j< Individual_Num; j++)
	   {  
		   float sum =0.0;
		   for (int m=0; m< Marker_Num; m++)
		   { 
			  	float temp1 = *(GMatrix_I+m*Individual_Num +i);
			  	float temp2 = *(GMatrix_I+m*Individual_Num +j); 
			  	sum += temp1 * temp2;			   
		   }
		   *(Kinship_Matrix_Out+i* Individual_Num+j) = sum; 		   
	   }
    }
}

void Golden_Kinship_Matrix_Add (float *Kinship_Matrix_Sub, float *Kinship_Matrix, long Matrix_Size)
{ 
	 for (long i = 0; i < Matrix_Size; i ++)
     Kinship_Matrix[i] = Kinship_Matrix[i] +  Kinship_Matrix_Sub[i];
}


void Golden_Kinship_Matrix_Normalize(float *Kinship_Matrix, long Matrix_Size, float Normalize_Rate) 
{  
   for (long i = 0; i < Matrix_Size; i ++)
      Kinship_Matrix[i] = Kinship_Matrix[i]/Normalize_Rate ;
}

void Golden_Matrix_AllValue_Set(float *Matrix, long Matrix_Size, float Value_Set) 
{   
   for (long i = 0; i < Matrix_Size; i ++)
       Matrix[i] = Value_Set;
}

int main(int argc, char* argv[])
{
   clock_t c_begin, c_end;
   c_begin=clock();
 
   // begin to calculate the time
   string Z_File_Name, W_File_Name, K_File_Name;   
   char Run_Mode;
   int Individual_Num, Marker_Num, MarkerPair_Block_Size;
   string OS_Path_Sep="//";
   long read_line_count=0;

   // Parse the command line for the inputting
   if(1==parse_cmd_line(argc, argv, Z_File_Name, W_File_Name, K_File_Name, Individual_Num, Marker_Num, MarkerPair_Block_Size, Run_Mode))
        return 1;
  
   //Open and Read the Geneotype Matrix .
	ifstream Z_File(Z_File_Name.c_str(), ios::in);
	ifstream W_File(W_File_Name.c_str(), ios::in);

    if(!Z_File.is_open())
	{
		cerr<< Z_File_Name <<" The 1st Genotype File Can't be accessed!"<<endl;
		return 1;
	}
	
	if(!W_File.is_open())
	{
		cerr<< W_File_Name <<" The 2nd Genotype File Can't be accessed!"<<endl;
		return 1;
	}
	
				
	if(Z_File && W_File)
    {
       float *Z_Matrix, *W_Matrix;
       string sLine;
       
       Z_Matrix = new float[Individual_Num*Marker_Num];
       
       int Line_Count=0;      
	   while(getline(Z_File, sLine))
	   {
		  if(sLine.empty()) ; // Ignore empty lines
		  else
		  {
            stringstream ss(sLine);
			vector <string> s_v;
			string item;
			char delim1 =','; 
			char delim2 ='\t';
		
			while(getline(ss, item, delim1))
			{
				s_v.push_back(item);
			}

			int Col_Num=s_v.size(); 
			if (Col_Num < Individual_Num)
			{
				s_v.clear(); 
				while(getline(ss, item, delim2))// try delim2;
			    {
				   s_v.push_back(item);
			    }
			    Col_Num=s_v.size(); 				
			}
			
			if(Col_Num != Individual_Num) 
			{
				cerr<< Z_File_Name <<"File Format (Indivual Num) is not right, Error at Line=" << Line_Count<<endl;
				if(Z_Matrix!=NULL) 
				{
				   delete []Z_Matrix; 
				   Z_Matrix =NULL;
			    }
		        return 1;
			}
			
			for (int i=0; i<Col_Num;i++)
			{
			   float value= atof(s_v.at(i).c_str());
			     Z_Matrix[Line_Count*Individual_Num+i] = value;
			}
          }
		  Line_Count++;
       }  
            
       if(Marker_Num != Line_Count)
       {
		   cerr<< Z_File_Name <<"File Format (Marker Num) is not Right, Total Read Line Num =" << Line_Count<<endl;
		   if(Z_Matrix!=NULL) 
		   {  
			  delete []Z_Matrix;
			  Z_Matrix =NULL;
		   }
		   return 1;
	   }
       
       Line_Count=0; 
       W_Matrix = new float[Individual_Num*Marker_Num];            
	   while(getline(W_File, sLine))
	   {
		  if(sLine.empty()) ; // Ignore empty lines
		  else
		  {
            stringstream ss(sLine);
			vector <string> s_v;
			string item;
			char delim1 =','; 
			char delim2 ='\t';
		
			while(getline(ss, item, delim1))
			{
				s_v.push_back(item);
			}

			int Col_Num=s_v.size(); 
			if (Col_Num < Individual_Num)
			{
				s_v.clear(); 
				while(getline(ss, item, delim2))// try delim2;
			    {
				   s_v.push_back(item);
			    }
			    Col_Num=s_v.size(); 				
			}
			
			if(Col_Num != Individual_Num) 
			{
				cerr<< W_File_Name <<"File Format (Indivudal Num) is not right, Error at Line=" << Line_Count<<endl;
				if(W_Matrix!=NULL) 
				{
					delete []W_Matrix;
					W_Matrix =NULL;
				}
		        return 1;
			}
			
			for (int i=0; i<Col_Num;i++)
			{
			   float value= atof(s_v.at(i).c_str());
			     W_Matrix[Line_Count*Individual_Num+i] = value;
			}
          }
		  Line_Count++;
       }
       
       if(Marker_Num != Line_Count)
       {
		   cerr<< W_File_Name <<"File Format (Marker Num) is not Right, Total Read Line Num =" << Line_Count<<endl;
		   if(W_Matrix!=NULL) 
		   {  
			  delete []W_Matrix;
			  W_Matrix =NULL;
		   }
		   return 1;
	   }
             
       float *G_Matrix, *G_Matrix_Tran, *KinshipMatrix_Sub, *KinshipMatrix;
       
       const long g_matrix_size =  Individual_Num*MarkerPair_Block_Size;
       const long k_matrix_size =  Individual_Num*Individual_Num;
       
       switch (Run_Mode)
       {
         case GPU_PARA :
            cudaMallocManaged(&G_Matrix, sizeof(float)*g_matrix_size);
            cudaMallocManaged(&G_Matrix_Tran, sizeof(float)*g_matrix_size);
            cudaMallocManaged(&KinshipMatrix_Sub, sizeof(float)*k_matrix_size);
            cudaMallocManaged(&KinshipMatrix, sizeof(float)*k_matrix_size);
            break;
         default :  
		    G_Matrix          = new float[g_matrix_size]; 
		    KinshipMatrix_Sub = new float[k_matrix_size];
		    KinshipMatrix     = new float[k_matrix_size];
		    break;
	   }
       	   
	   dim3 thread2ds(TILE_DIM,TILE_DIM);
       dim3 grid2ds((Individual_Num+TILE_DIM-1)/TILE_DIM, (MarkerPair_Block_Size+TILE_DIM-1)/TILE_DIM);
       dim3 grid2ds1((Individual_Num+TILE_DIM-1)/TILE_DIM, (Individual_Num+TILE_DIM-1)/TILE_DIM);
       dim3 thread1ds(Block_Size);
	   dim3 grid1ds((Individual_Num*Individual_Num)/Block_Size);
	   
	   int MairPair_Count =0;
	   for (int Marker_Z=0; Marker_Z < Marker_Num ; Marker_Z++)
	   for (int Marker_W=Marker_Z+1; Marker_W < Marker_Num ; Marker_W++)
	   {
		   for (int index =0; index < Individual_Num; index ++)
			 G_Matrix[MairPair_Count*Individual_Num+index] = Z_Matrix[Marker_Z*Individual_Num+index] * W_Matrix[Marker_W*Individual_Num+index];
			 
		   MairPair_Count ++;			
			
		   if (MairPair_Count < MarkerPair_Block_Size)
			  continue;
		   else  // Process the full Marker_Pair Block 
		   { 
			  switch(Run_Mode)
			  {
			    case GPU_PARA :
			       // begin to call the GPU program to calculate the partial kinship matrix			  
                   Genotype_Matrix_Transpose<<<grid2ds, thread2ds >>>(G_Matrix_Tran, G_Matrix, Individual_Num, MarkerPair_Block_Size);
                   cudaDeviceSynchronize();// Wait for GPU to finish before accessing on host
             
                   Kinship_Matrix_Cal<<<grid2ds1, thread2ds >>>(G_Matrix_Tran, G_Matrix, KinshipMatrix_Sub, Individual_Num, MarkerPair_Block_Size);  
                   cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
			 
			       // add the block kinship matrix into the all kinship matrix
			       Kinship_Matrix_Add <<<grid1ds, thread1ds>>>(KinshipMatrix_Sub,  KinshipMatrix, k_matrix_size); 		  
		           cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
					 
			       Matrix_AllValue_Set <<<grid1ds, thread1ds>>> (G_Matrix, g_matrix_size, 0.0);
			       cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host		
			       break;	
			    case CPU_GOLDEN:
			       Golden_Kinship_Matrix_Cal(KinshipMatrix_Sub, G_Matrix,  Individual_Num,  MarkerPair_Block_Size);
			       Golden_Kinship_Matrix_Add(KinshipMatrix_Sub, KinshipMatrix, k_matrix_size);
			       Golden_Matrix_AllValue_Set(G_Matrix, g_matrix_size, 0.0); 
			       break;
			    default :
			       break;  			         
			  }
			  MairPair_Count =0; 
		  }	   		     
	   }	  
       
       if(MairPair_Count>0) //  At the rear part of the residual marker pair lines 
       { 
		   
		   switch(Run_Mode)
		   {
			 case GPU_PARA :  
		       
                Genotype_Matrix_Transpose<<<grid2ds, thread2ds >>>(G_Matrix_Tran, G_Matrix, Individual_Num, MarkerPair_Block_Size);
                cudaDeviceSynchronize();// Wait for GPU to finish before accessing on host
                        
                Kinship_Matrix_Cal<<<grid2ds1, thread2ds >>>(G_Matrix_Tran, G_Matrix, KinshipMatrix_Sub, Individual_Num,  MarkerPair_Block_Size);  
                cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
			 
		        // add the block kinship matrix into the all kinship matrix
		        Kinship_Matrix_Add <<<grid1ds, thread1ds>>>(KinshipMatrix_Sub,  KinshipMatrix, k_matrix_size); 
		        cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
		        break;
		        
		     case CPU_GOLDEN:
			    Golden_Kinship_Matrix_Cal(KinshipMatrix_Sub, G_Matrix,  Individual_Num,  MarkerPair_Block_Size);
			    Golden_Kinship_Matrix_Add(KinshipMatrix_Sub, KinshipMatrix, k_matrix_size);
			    Golden_Matrix_AllValue_Set(G_Matrix, g_matrix_size, 0.0);   
		        break;
		        
			  default:
			    break;   
		   }		   
		   MairPair_Count =0;  // Reset the MairPair_Count to 0 
	   }
	   
	   // Begin to calculate the Normalized ratio 
	   float Normalize_Rate =0.0;
	   for (int i_index =0; i_index < Individual_Num; i_index++)
	   Normalize_Rate += KinshipMatrix[i_index*Individual_Num+i_index];
	  
	   Normalize_Rate =Normalize_Rate/Individual_Num;
	   
	   switch(Run_Mode)
	   {
	      case GPU_PARA : 	      
	        Kinship_Matrix_Normalize <<<grid1ds, thread1ds>>>(KinshipMatrix, k_matrix_size, Normalize_Rate) ;
	        cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
	        break;
	      case CPU_GOLDEN:
	        Golden_Kinship_Matrix_Normalize(KinshipMatrix, k_matrix_size, Normalize_Rate);
	        break;
	      default :
	        break;
	   }
	   
	   //Output the kinship matrix
	   ofstream Kinship_File; // used to output the calculated kinship matrix 
	   Kinship_File.open(K_File_Name.c_str());
	   if(Kinship_File.is_open())
	   {
			if(Run_Mode <READ_THRU)
			{
			   char delim ='\t';			
			   for (int i_individual=0; i_individual < Individual_Num; i_individual++)
			   {
			     Kinship_File<<KinshipMatrix[i_individual*Individual_Num]; 
			     for (int j_individual=1; j_individual < Individual_Num; j_individual++)
			     {
			        Kinship_File<< delim<< KinshipMatrix[i_individual*Individual_Num + j_individual];
			     }
			     Kinship_File<<endl;  // end line			   
			   }
		    }
		    else
		    {
				Kinship_File << "Read through, Read_Line_Count is:" << read_line_count << endl;
			}
		}
		else
		{
			cerr <<"Error open the result file for the output Kinship Matrix!"<<endl;
		}

		Kinship_File.close();
	   
	    // Free the allocated matrix 	
	    switch (Run_Mode)
	    {  
		   case GPU_PARA : 
              cudaFree(G_Matrix);
              cudaFree(G_Matrix_Tran);
              cudaFree(KinshipMatrix_Sub);
              cudaFree(KinshipMatrix);  
              break;
           default:   	    
		      delete []G_Matrix; 
		      delete []KinshipMatrix_Sub;
		      delete []KinshipMatrix;
		      break;
		}     
	} 

   c_end =clock();
   double elapse_time = double (c_end-c_begin)/CLOCKS_PER_SEC; 
   cout << "Hello, the elapse time is " <<  elapse_time << " seconds" << endl;
  
}

void test_matrix_transpose() 
{
   /* The following are GPU interface for testing the GPU version for Matrix Transpose at arbitrary dimension*/ 
  
   int Matrix_Width  =10000;
   int Matrix_Height =50000;  
   dim3 threads(TILE_DIM,TILE_DIM);
   dim3 grids((Matrix_Width+TILE_DIM-1)/TILE_DIM, (Matrix_Height+TILE_DIM-1)/TILE_DIM);
   
   const long mem_size = sizeof(float)*Matrix_Width*Matrix_Height;
   float *GMatrix_O, *GMatrix_I;
   cudaMallocManaged(&GMatrix_O, mem_size);
   cudaMallocManaged(&GMatrix_I, mem_size);
  
   // initalize host data
   for(long i = 0; i < (Matrix_Width*Matrix_Height); ++i)
   GMatrix_I[i] = (float) i;
     
   Genotype_Matrix_Transpose<<<grids, threads >>>(GMatrix_O, GMatrix_I, Matrix_Width, Matrix_Height);
  
   // Wait for GPU to finish before accessing on host
   cudaDeviceSynchronize();
  
   
   float maxError = 0.0f;
   for (int i = 0; i < Matrix_Height; i++)
   { 
	   for (int j = 0; j < Matrix_Width; j++)
	   {
		   maxError = fmax(maxError, fabs(GMatrix_I[i*Matrix_Width+j]-GMatrix_O[j*Matrix_Height+i]));
		   if(maxError>0.1)
		     cout << "i=" <<i << "j=" <<j << "GMatrix_I(i,j)="  << GMatrix_I[i*Matrix_Width+j] << "GMatrix_O(j,i)=" << GMatrix_O[j*Matrix_Height+i]<< endl;
	   }
   }
     
  std::cout << "Max error: " << maxError << std::endl;
   
  cudaFree(GMatrix_O);
  cudaFree(GMatrix_I);
  return ; 	
}

void test_kinship()
{
	  /* The following are GPU interface for testing the GPU version for kinship matrix calculating by multiplying the transpose of genotype Matrix and genotype matrix at arbitrary dimension*/ 
   
   int Marker_Num =1000;
   int Individual_Num =2500;
    
   dim3 threads(TILE_DIM,TILE_DIM);
   dim3 grids((Individual_Num+TILE_DIM-1)/TILE_DIM, (Marker_Num+TILE_DIM-1)/TILE_DIM);
   
   const long genotype_matrix_size = sizeof(float)*Marker_Num*Individual_Num;
   const long kinship_matrix_size = sizeof(float)* Individual_Num*Individual_Num;
   float *Genotype_Matrix_T,  *Genotype_Matrix,  *Kinship_Matrix;
   cudaMallocManaged(&Genotype_Matrix_T, genotype_matrix_size);
   cudaMallocManaged(&Genotype_Matrix, genotype_matrix_size);
   cudaMallocManaged(&Kinship_Matrix, kinship_matrix_size);
    
   // initalize host data
   for(long i = 0; i < (Marker_Num*Individual_Num); ++i)
   Genotype_Matrix[i] = (float) 1.0*i/(Marker_Num);
     
   Genotype_Matrix_Transpose<<<grids, threads >>>(Genotype_Matrix_T, Genotype_Matrix, Individual_Num, Marker_Num);
  
   // Wait for GPU to finish before accessing on host
   cudaDeviceSynchronize();
   
      
   float maxError = 0.0f;
   for (int i = 0; i < Marker_Num; i++)
   { 
	   for (int j = 0; j < Individual_Num; j++)
	   {
		   maxError = fmax(maxError, fabs(Genotype_Matrix[i*Individual_Num+j]-Genotype_Matrix_T[j*Marker_Num+i]));
		   if(maxError>0.1)
		     cout << "i=" <<i << "j=" <<j << "Genotype_Matrix(i,j)="  << Genotype_Matrix[i*Individual_Num+j] << "Genotype_Matrix_T(j,i)=" << Genotype_Matrix_T[j*Marker_Num+i]<< endl;
	   }
   }
     
  std::cout << "Matrix Transpose Max error: " << maxError << std::endl;
  
  dim3 grids1((Individual_Num+TILE_DIM-1)/TILE_DIM, (Individual_Num+TILE_DIM-1)/TILE_DIM);
  Kinship_Matrix_Cal<<<grids1, threads >>>(Genotype_Matrix_T, Genotype_Matrix, Kinship_Matrix, Marker_Num, Individual_Num);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  std::cout <<"Kinship_Matrix[0,0]="<< Kinship_Matrix[0] <<"Kinship_Matrix[0,1]=" <<Kinship_Matrix[1] << "Kinship_Matrix[0,1023]="<< Kinship_Matrix[1023] <<"Kinship_Matrix[1,1]=" <<Kinship_Matrix[1024]<< std::endl; 
 
  cudaFree(Genotype_Matrix_T);
  cudaFree(Genotype_Matrix);
  cudaFree(Kinship_Matrix); 
  return;
}

void test_mis()
{
/*	  std::cout << "g_matrix(0,0)=" <<  G_Matrix[0]<< std::endl;
			 cudaMemset(G_Matrix, 0, g_mem_size*sizeof(float));
			 cudaMemset(KinshipMatrix_Sub, 0, k_mem_size*sizeof(float));
	  std::cout << "Line_Count= " << Line_Count << std::endl;
			 std::cout << "g_matrix(0,0)=" <<  G_Matrix[0]<< std::endl;
			 std::cout << "kmatrix_sub(0,0)=" <<  KinshipMatrix_Sub[0]<< std::endl;
			 std::cout << "kmatrix(0,0)=" <<  KinshipMatrix[0]<< std::endl;
			 std::cout << "Line_Count= " << Line_Count << std::endl;	  
		   std::cout << "kmatrix_sub( 0,0)=" <<  KinshipMatrix_Sub[0]<< std::endl;
		   std::cout << "kmatrix(0,0)=" <<  KinshipMatrix[0]<< std::endl;
		
*/

  /*   dim3 thread2ds(TILE_DIM,TILE_DIM);
       dim3 grid2ds((Individual_Num+TILE_DIM-1)/TILE_DIM, (MarkerBlock_Size+TILE_DIM-1)/TILE_DIM);*/
    
                   /*dim3 grid2ds1((Individual_Num+TILE_DIM-1)/TILE_DIM, (Individual_Num+TILE_DIM-1)/TILE_DIM);*/  
                     /*dim3 thread1ds(Block_Size);
			       dim3 grid1ds((Individual_Num*Individual_Num)/Block_Size);*/ 
			       
			       /* dim3 thread2ds(TILE_DIM,TILE_DIM);
                dim3 grid2ds((Individual_Num+TILE_DIM-1)/TILE_DIM, (MarkerBlock_Size+TILE_DIM-1)/TILE_DIM);*/
                
                 /*  dim3 grid2ds1((Individual_Num+TILE_DIM-1)/TILE_DIM, (Individual_Num+TILE_DIM-1)/TILE_DIM);*/
                   /*   dim3 thread1ds(Block_Size);
		        dim3 grid1ds((Individual_Num*Individual_Num)/Block_Size);*/
		        /*  dim3 thread1ds(Block_Size);
	        dim3 grid1ds((Individual_Num*Individual_Num)/Block_Size);*/
       
  return;
}
