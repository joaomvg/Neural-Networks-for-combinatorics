#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <ctime>

//Neural Network class for prime/non-prime numbers
using namespace std;

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

void create_ptr(double **x,int d){
  x=(double **) malloc(d * sizeof(double*));
  for(int i=0;i<d;i++){
    x[i]=(double *) malloc(1*sizeof(double));
  }
}

double Random(){//random number [0,1]
  return 10*(double(rand())/(RAND_MAX))-5;
}

class NN{
  int n; //it should be the dimension of the input array; to be determined according to the problem
  int nn1; //number of neurons in layer1
  int nn2; //number of neurons in layer2
  double t;

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
        t=1; //learning rate

        output=(double *) malloc(d*sizeof(double));
        y=(double *) malloc(d*sizeof(double));
        for(int i=0;i<d;i++){
          y[i]=y_in[i];
        }
        create_ptr(x,d);

        for(int i=0;i<d;i++){
          for(int j=0;j<n;j++){
            x[i][j]=x_in[i][j];
          }
        }

        create_ptr(weights1,nn1);

        for(int i=0;i<nn1;i++){
          for(int j=0;j<n;j++){
            weights1[i][j]=Random(); //intialise weights1
          }
        }

        b1=(double *) malloc(nn1*sizeof(double));
        b2=Random();
        weights2=(double *) malloc(nn2*sizeof(double));
        create_ptr(layer1,nn1);
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
          b1[i]-=2*t*(output[k]-y[k])*output[k]*(1-output[k])*weights2[i]*layer1[k][i]*(1-layer1[k][i]);
        }
      }

      for(int i=0;i<nn2;i++){
        for(int j=0;j<nn1;j++){
          for(int k=0;k<d;k++){
            weights1[i][j]-=2*t*(output[k]-y[k])*output[k]*(1-output[k])*weights2[i]*layer1[k][i]*(1-layer1[k][i])*x[k][j];
          }
        }
      }

        for(int i=0;i<nn2;i++){
          for(int k=0;k<d;k++){
            weights2[i]-=2*t*(output[k]-y[k])*output[k]*(1-output[k])*layer1[k][i];
          }
        }


        for(int k=0;k<d;k++){
            b2-=2*t*(output[k]-y[k])*output[k]*(1-output[k]);
          }


    }

    double Loss(){
        double l=0;
        for(int i=0;i<d;i++){
          l+=pow(output[i]-y[i],2)/d;
        }
        return l;

    }

    void Free(){
      for(int i=0;i<nn1;i++){
        free(weights1[i]);
      }
      free(weights1);

      free(weights2);
      for(int i=0;i<d;i++){
        free(layer1[i]);
      }
      free(layer1);
      free(b1);
      free(output);
      for(int i=0;i<d;i++){
        free(x[i]);
      }
      free(x);
      free(y);
    }

};



int main(){
  srand(time(NULL));
  double **x_in,*y_in;
  int num=5;
  ifstream f;
  ofstream f_out;

  f_out.open("loss.dat");
  f.open("primes.dat");

  create_ptr(x_in,num);
  y_in=(double *) malloc(num * sizeof(double));

  int i=0;

  while(i<num){
    f>>x_in[i][0];
    x_in[i][0]=x_in[i][0];
    f>>y_in[i];
    i+=1;
  }

  f.close();
  int iterations=100000;

  NN net(1,3,num,x_in,y_in);

  i=0;
  while (i<iterations){
    net.FF();
    net.BB();
    f_out<<i<<" "<<net.Loss()<<endl;
    i+=1;
  }
  f_out.close();
  for(int i=0;i<num;i++){
    printf("y_true=%lf, y_pred=%lf\n",y_in[i], net.output[i]);
  }

  net.Free();
  for(int i=0;i<num;i++){
    free(x_in[i]);
  }
  free(x_in);
  free(y_in);

  return 0;
}
