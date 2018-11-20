#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <afxtempl.h>
#include "Layer.h"

#define INPUT_NEURONS		16
#define HIDDEN_NEURONS		32 
#define OUTPUT_NEURONS		26

#define sqr(x)	((x) * (x))

const int SOFTMAX = 1;
const int HIDDEN_LAYER_NUMBER = 2;
const int NUMBER_OF_PATTERNS = 20000;
const int NUMBER_OF_TRAINING_PATTERNS = 16000;
const int NUMBER_OF_TEST_PATTERNS = 4000;

const int DEFAULT_MAX_EPOCHS = 100;
const double SIGMOID_LEARNING_RATE = 0.01;
const double RELU_LEARNING_RATE = 0.0001;
const double TANH_LEARNING_RATE = 0.0005;

struct Letter_S {
	char symbol;
	double O[OUTPUT_NEURONS];
	double X[INPUT_NEURONS];
	void reset() {
		symbol = 0;
		memset(X, 0, sizeof(double)*INPUT_NEURONS);
		if (SOFTMAX == 1)
		{
			memset(O, 0, sizeof(double)*OUTPUT_NEURONS);
		}
		else
		{
			for (int i = 0; i < OUTPUT_NEURONS; i++)
				O[i] = 0.1;
		}
	}
};

struct Assess_S {
	double trainSSE;
	double trainMse;
	double trainRatio; //percent of good classification
	double testSSE;
	double testMse;
	double testRatio;
	int    confusionMatrix[OUTPUT_NEURONS][OUTPUT_NEURONS];
};

class NeuralNetwork
{
public:
	NeuralNetwork();
    void initialise();
    int saveWeights(const char* fileName);
    int loadWeights(const char* fileName);

	double preTrainNetwork(int epochsPretrain, BOOL lastLayer);
    int train(int numberOfEpochs, double learningRate, BOOL pretrain, int epochsPretrain, BOOL lastLayer);
	double* test(Letter_S* testPattern);
	Assess_S* assess();
	char classify(double* vals);
	
	void loadTrainPatterns(const char* fileName);
	void shuffleTrainData();
	void loadTestPatterns(const char* fileName);
	Letter_S* parsePattern(CString& pattern, Letter_S* letter);

	void setActivationFunction(ActivationFunction af);
	ActivationFunction getActivationFunction() { return m_af; }
	void setLearningRate(double rate);
private:
	void makeCache(CArray<double*>& cache, int number, int itemSize);
	void clearCache(CArray<double*>& cache);
	void loadPatterns(const char* fileName, CArray<Letter_S>& patternSet);
	void backPropagate();
	void setInput(double* input);
	void feedForward();
	
public:
	CArray<Letter_S> m_train;
	CArray<Letter_S> m_test;
private:
	Layer* m_pInput;
	Layer* m_pHidden0;
	Layer* m_pHidden1;
	Layer* m_pOutput;

	double m_learningRate;
	int m_maxEpochs;
	int m_accEpochs;   //accumulated epochs
	Assess_S m_assess;
	BOOL m_preTrained;

	ActivationFunction m_af;
};

#endif // NEURALNETWORK_H
