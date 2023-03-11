//relevant headers
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <numeric>

using namespace std;

//defining the CPPNN class

class CPPNN {
private:
    // number of nodes in layers
    int in_layers, h1_layers,h2_layers, out_layers;
    //weight matrices
    vector<vector<double>> W1, W2, W3;
    //internal vectors of the nn
    vector<double> o1, o2, o3, o4, o5, o6, error, o6delta, o4error, o4delta, o2error, o2delta;



    // creating a function to generate weight matrices with randomized numbers
    vector<vector<double>> init_weights(int rows, int columns) {
        vector<vector<double>> matrix(rows, vector<double>(columns));
        default_random_engine generator;
        // Random numbers are generated using normal distribution
        normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                matrix[i][j] = distribution(generator);
            }
        }
        return matrix;
    }

public:

    // The Neural Network consists 2 hidden layers with each consisting 6 neurons, input layer consists 3 neurons and output has 1 neuron
    CPPNN() {
        int in_layers = 3;
        int h1_layers = 6;
        int h2_layers = 6;
        int out_layers = 1;

        // creating weight matrices for each layers
        W1 = init_weights(in_layers, h1_layers);
        W2 = init_weights(h1_layers, h2_layers);
        W3 = init_weights(h2_layers, out_layers);
    }
    //defining sigmoid activation function

    double sigmoid(double x){

        return 1.0 / (1.0 + exp(-x));
    }

    //defining sigmoid_prime function for back propagation

    double sigmoidprime(double x){
        return x*(1-x);
    }


    //defining the feed forward function

    double feedforward(vector<double> X){

        // input to Hidden layer 1
        for (int i = 0; i < h1_layers; i++) {
            o1[i] = 0;
            for (int j = 0; j < in_layers; j++) {
                o1[i] += X[j] * W1[j][i];
            }
            o2[i] = sigmoid(o1[i]);
        }

        // HL1 to HL2
        for (int i = 0; i < h2_layers; i++) {
            o3[i] = 0;
            for (int j = 0; j < h1_layers; j++) {
                o3[i] += o2[j] * W2[j][i];
            }
            o4[i] = sigmoid(o3[i]);
        }

        // HL2 to output
        for (int i = 0; i < out_layers; i++) {
            o5[i] = 0;
            for (int j = 0; j < h2_layers; j++) {
                o5[i] += o4[j] * W3[j][i];
            }
            o6[i] = sigmoid(o5[i]);
        }

        return o6[0];

    }


    //defining the backpropagation function
    void backpropagation(vector<double> X, vector<double> Y, vector<double> o6){

        for (int i = 0; i < out_layers; i++) {
            error[i] = Y[i] - o6[i];
        }

        // applying the derivative of the Sigmoid activation to the error
        for (int i = 0; i < out_layers; i++) {
            o6delta[i] = error[i] * sigmoidprime(o6[i]);
        }

        // finding the contribution of the last layer to the error
        for (int i = 0; i < h2_layers; i++) {
            o4error[i] = 0.0;
            for (int j = 0; j < out_layers; j++) {
                o4error[i] += o6delta[j] * W3[i][j];
            }
        }

        // applying the derivative of the o4 to o4 error
        for (int i = 0; i < h2_layers; i++) {
            o4delta[i] = o4error[i] * sigmoidprime(o4[i]);
        }

        // finding the contribution of the second layer to the error
        for (int i = 0; i < h1_layers; i++) {
            o2error[i] = 0.0;
            for (int j = 0; j < h2_layers; j++) {
                o2error[i] += o4delta[j] * W2[i][j];
            }
        }

        // applying the derivative of the o2 to o2 error
        for (int i = 0; i < h1_layers; i++) {
            o2delta[i] = o2error[i] * sigmoidprime(o2[i]);
        }

        // adjusting first set (inputLayer --> h1) weights
        for (int i = 0; i < in_layers; i++) {
            for (int j = 0; j < h1_layers; j++) {
                W1[i][j] += X[i] * o2delta[j];
            }
        }

        // adjusting second set (h1 --> h2) weights
        for (int i = 0; i < h1_layers; i++) {
            for (int j = 0; j < h2_layers; j++) {
                W2[i][j] += o2[i] * o4delta[j];
            }
        }

        // adjusting third set (h2 --> output) weights
        for (int i = 0; i < h2_layers; i++) {
            for (int j = 0; j < out_layers; j++) {
                W3[i][j] += o4[i] * o6delta[j];
            }
        }

    }


    //training function

    void trainNN(vector<double> X,vector<double> Y) {
    // forward feeding
        o6[0] = feedforward(X);
        // back propagation
        backpropagation(X, Y, o6);
}


















};



int main() {
    CPPNN sampleNN;
    int trainingEpochs = 100;
    vector<double> X;
    vector<double> Y;

    for (int i = 0; i < trainingEpochs; i++) {
        cout << "Epoch # " << i << "\n";
        cout << "Network Input : \n";
        for (const auto& x : X) {
            cout << x << " ";
        }
        cout << "\nExpected Output : \n";
        for (const auto& y : Y) {
            cout << y << " ";
        }
        cout << "\nActual Output : \n";
        for (const auto& out : sampleNN.feedforward(X)) {
            cout << out << " ";
        }

        // mean sum squared loss
        double Loss = 0.0;
        for (int j = 0; j < Y.size(); j++) {
            Loss += pow(Y[j] - sampleNN.feedforward(X)[j], 2);
        }
        Loss /= Y.size();
        //sampleNN.saveSumSquaredLossList(i, Loss);
        cout << "\nSum Squared Loss: \n" << Loss << "\n\n";
        sampleNN.trainNN(X, Y);
    }
    //sampleNN.saveWeights();
    return 0;
}
