#include <iostream>
#include <cstdlib>
#include <cmath>

//define Neural Network classes



double sigmoid(double x){
  return 1/(1+exp(-x));
}

double dot(double *x, double *y, int dim){
    double s=0;
    for(int i=0; i<dim;i++){
      s+=x[i]*y[i];
    }

    return s;
}

double Random(){//random number [0,1]
  return double(rand())/RAND_MAX;
}

class NN{
  int n; //it should be the dimension of the array; to be determined according to the problem
  int nn1; //number of neurons in layer1
  int nn2; //number of neurons in layer2

  public:
    double **weights1;
    double *weights2;
    double *b1; //bias1
    double b2; //bias2
    double **layer1;
    double *output;
    double **x;
    double *y;
    int d;

    NN(int m1,int m2,int dim, double **x_in, double *y_in){ //constructor
        n=m1;
        nn1=m2;
        nn2=nn1;
        d=dim;

        output=(double *) malloc(d*sizeof(double));

        x=x_in;
        y=y_in;

        weights1= (double **) malloc(nn1*sizeof(double *)); //number of rows=nn1
        for(int i=0;i<nn1;i++){
          weights1[i]=(double *) malloc(n*sizeof(double));
        }

        for(int i=0;i<nn1;i++){
          for(int j=0;j<n;j++){
            weights1[i][j]=Random(); //intialise weights1
          }
        }

        b1=(double *) malloc(nn1*sizeof(double));
        for(int i=0;i<nn1;i++){
          b1[i]=Random();
        }

        b2=Random();

        weights2= (double *) malloc(nn2*sizeof(double));
        for(int i=0;i<nn2;i++){
          weights2[i]=Random();
        }

        layer1=(double **)malloc(d*sizeof(double *));
        for(int i=0;i<d;i++){
          layer1[i]=(double *) malloc(nn1*sizeof(double));
        }

    }

    void FF(){//FeedForward function
      for(int i=0;i<d;i++){
        for(int j=0;j<nn1;j++){
          layer1[i][j]=sigmoid(dot(weights1[j],x[i],n)+b1[j]);
        }
        output[i]=sigmoid(dot(layer1[i],weights2,n)+b2);
      }
    }

    void BB(){
      for(int i=0;i<nn1;i++){
        for(int k=0;k<d;k++){
          b1[i]-=2*(output[k]-y[k])*output[k]*(1-output[k])*weights2[i]*layer1[k][i]*(1-layer1[k][i]);
        }
      }

      for(int i=0;i<nn2;i++){
        for(int j=0;j<nn1;j++){
          for(int k=0;k<d;k++){
            weights1[i][j]-=2*(output[k]-y[k])*output[k]*(1-output[k])*weights2[i]*layer1[k][i]*(1-layer1[k][i])*x[k][j];
          }
        }
      }

        for(int i=0;i<nn2;i++){
          for(int k=0;k<d;k++){
            weights2[i]-=2*(output[k]-y[k])*output[k]*(1-output[k])*layer1[k][i];
          }
        }


        for(int k=0;k<d;k++){
            b2-=2*(output[k]-y[k])*output[k]*(1-output[k]);
          }


    }

};



int main(){
 srand(time(NULL));
  double **x_in;
  double x_aux[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};


  x_in=(double **) malloc(4*sizeof(double *));
  for(int i=0;i<4;i++){
    x_in[i]=(double *) malloc(3*sizeof(double));
  }

  for(int i=0;i<4;i++){
    for(int j=0;j<3;j++){
      x_in[i][j]=x_aux[i][j];
    }
  }

  double y_in[4]={0,1,1,0};
  int iterations=1500;

  NN net(3,4,4,x_in,y_in);

  int i=0;
  while (i<iterations) {
    net.FF();
    net.BB();
    i+=1;
  }


for(int j=0;j<4;j++){
  printf("%lf\n",net.output[j]);
}

  return 0;
}
