#include <iostream>
#include <cstdlib>
#include <cmath>

/*define Neural Network classes
- 3-Layer Net w3Sigmoid(w2.Sigmoid(w1.x+b1)+b2)
- w3=weights3; w2=weights2; w1=weights1
- b1,b2= biases

- Using sigmoid function */

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

double Random(){//random number [0,1]
  return double(rand())/RAND_MAX;
}

class NN{
  int n; //it should be the dimension of the input array; to be determined according to the problem
  int nn1; //number of neurons in layer1
  int nn2; //number of neurons in layer2

  public:
    double **weights1;
    double **weights2;
    double *weights3;
    double *b1; //bias1
    double *b2; //bias2
    double **layer1;
    double **layer2;
    double *output;
    double **x;
    double *y;
    int d;

    NN(int m1,int m2,int dim, double **x_in, double *y_in){ //constructor
        n=m1;
        nn1=m2;
        nn2=nn1;
        d=dim;

        x=(double **) malloc(d*sizeof(double *));
        for(int i=0;i<d;i++){
          x[i]=(double *) malloc(n*sizeof(double));
        }

        for(int i=0;i<d;i++){
          for(int j=0; j<n;j++){
            x[i][j]=x_in[i][j];
          }
        }

        y=(double *) malloc(d*sizeof(double *));
        for(int i=0;i<d;i++){
          y[i]=y_in[i];
        }

        output=(double *) malloc(d*sizeof(double));

        layer2=(double **) malloc(d*sizeof(double));
        for(int i=0;i<d;i++){
          layer2[i]=(double *) malloc(nn2*sizeof(double));
        }

        layer1=(double **)malloc(d*sizeof(double *));
        for(int i=0;i<d;i++){
          layer1[i]=(double *) malloc(nn1*sizeof(double));
        }

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
          b1[i]=Random(); //initialise bias1
        }



        weights2= (double **) malloc(nn2*sizeof(double *)); //number of rows=nn1
        for(int i=0;i<nn2;i++){
          weights2[i]=(double *) malloc(nn1*sizeof(double));
        }

        for(int i=0;i<nn2;i++){
          for(int j=0;j<nn1;i++){
            weights2[i][j]=Random(); //initialise weights2
          }
        }

        b2=(double *) malloc(nn2*sizeof(double));
        for(int i=0;i<nn2;i++){
          b2[i]=Random();//initialise b2
        }

        weights3=(double *) malloc(nn2*sizeof(double));
        for(int i=0;i<nn2;i++){
          weights3[i]=Random();
        }

    }


    void FF(){//FeedForward function

      //1st layer
      for(int i=0;i<d;i++){ //d goes through data available
        for(int j=0;j<nn1;j++){
          layer1[i][j]=sigmoid(dot(weights1[j],x[i],n)+b1[j]);
        }
      }
      //2nd layer
      for(int i=0;i<d;i++){
        for(int j=0;j<nn2;j++){
          layer2[i][j]=sigmoid(dot(weights2[j],layer1[i],nn1)+b2[j]);
        }
      }

      //3rd layer

      for(int i=0;i<d;i++){
        output[i]=dot(weights3,layer2[i],nn2);
      }
    }

    void BP(){//BackPropagation

      for(int i=0;i<nn1;i++){
        for(int k=0;k<d;k++){
          for(int j=0;j<nn2;j++){
            b1[i]-=2*(output[k]-y[k])*weights3[j]*layer2[k][j]*(1-layer2[k][j])*weights2[j][i]*layer1[k][i]*(1-layer1[k][i]);
        }
      }
    }

      //weights1
      for(int i=0;i<nn1;i++){
        for(int j=0;j<n;j++){
          for(int l=0;l<nn2;l++){
            for(int k=0;k<d;k++){
              weights1[i][j]-=2*(output[k]-y[k])*weights3[l]*layer2[k][l]*(1-layer2[k][l])*weights2[l][i]*layer1[k][i]*(1-layer1[k][i])*x[k][j];
            }
          }
        }
      }

      //weights2
        for(int i=0;i<nn2;i++){
          for(int j=0;j<nn1;j++){
            for(int k=0;k<d;k++){
              weights2[i][j]-=2*(output[k]-y[k])*weights3[i]*layer2[k][i]*(1-layer2[k][i])*layer1[k][j];
            }
          }
        }

        //b2
        for(int i=0;i<nn2;i++){
          for(int k=0;k<d;k++){
            b2[i]-=2*(output[k]-y[k])*weights3[i]*layer2[k][i]*(1-layer2[k][i]);
          }
        }

        //weights3
        for(int i=0;i<nn2;i++){
          for(int k=0;k<d;k++){
            weights3[i]-=2*(output[k]-y[k])*layer2[k][i];
          }
        }
    }

};



int main(){
 srand(time(NULL));
  double **x_in;
  int dim;
  int n;
  int iterations=1500;
  int nn1=4; //play with this number. should be of the order of dim

  cin>>dim; //input dimension of data
  cin>>n; //input dimension of input array (depends on the topological data passed into the net, still experimenting)

  x_in=(double **) malloc(dim*sizeof(double *));
  for(int i=0;i<4;i++){
    x_in[i]=(double *) malloc(n*sizeof(double));
  }

  //First part is to test the NET using some data points. For the moment introduce it by hand
  for(int i=0;i<dim;i++){
    cout<<"new point"<<endl;
    for(int j=0;j<n;j++){
      cin>>x_in[i][j];
    }
  }

  cout<<"x input done!"<<endl;

  //output y
  double *y_in;


  y_in=(double *) malloc(dim*sizeof(double));
  for(int i=0;i<dim;i++){
      cin>>y_in[i];
  }

  cout<<"output y done!"<<endl;

  NN net(n,nn1,dim,x_in,y_in);

  int i=0;
  while (i<iterations) {
    net.FF();
    net.BP();
    i+=1;
  }


  for(int i=0;i<dim;i++){
    cout<<"y_in="<<y_in[i]<<"..."<<"y_pred="<<net.output[i]<<endl; //to compare y_in with y_prediction
  }

  free(y_in);
  for(int i=0;i<4;i++){
    free(x_in[i]);
  }
  free(x_in);

  return 0;
}
