#include "stdafx.h"
#include "NeuralNetwork.h"
#include "SoftmaxLayer.h"
#include "Autoencoder.h"

///////////////////////////////////////////////////////////////////
NeuralNetwork::NeuralNetwork()
{
}

void NeuralNetwork::initialise()
{
	memset(&m_assess, 0, sizeof(Assess_S));
	m_learningRate = RELU_LEARNING_RATE;
	m_maxEpochs = DEFAULT_MAX_EPOCHS;
	m_accEpochs = 0;
	m_af = ActivationFunction::AF_RELU;

	m_train.RemoveAll();
	m_test.RemoveAll();

	/* Seed the random number generator */
	srand((unsigned int)time(NULL));
	m_pInput = new Layer(0, INPUT_NEURONS, HIDDEN_NEURONS);
	if (HIDDEN_LAYER_NUMBER == 1)
	{
		m_pHidden0 = new Layer(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS);
		m_pHidden1 = nullptr;
	}
	else
	{
		m_pHidden0 = new Layer(INPUT_NEURONS, HIDDEN_NEURONS, HIDDEN_NEURONS);
		m_pHidden1 = new Layer(HIDDEN_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS);
	}
	if (SOFTMAX == 1)
		m_pOutput = new SoftmaxLayer(HIDDEN_NEURONS, OUTPUT_NEURONS, 0);
	else
		m_pOutput = new Layer(HIDDEN_NEURONS, OUTPUT_NEURONS, 0);

	m_preTrained = FALSE;
}

void NeuralNetwork::setActivationFunction(ActivationFunction af) {
	Neuron::setActivationFunction(af);
	m_af = af;
}

int NeuralNetwork::saveWeights(const char* fileName) {
	FILE* wfile = fopen(fileName, "w");
	if (wfile == nullptr) {
		printf("NeuralNetwork::saveWeights can not open file.");
		return -1;
	}
	//save network structure
	fprintf(wfile, "%d %d ", m_pInput->m_N, m_pHidden0->m_N);
	if (m_pHidden1 != nullptr)
		fprintf(wfile, "%d ", m_pHidden1->m_N);
	fprintf(wfile, "%d ", m_pOutput->m_N);
	fprintf(wfile, "%d\n", m_af);
	fprintf(wfile, "%d %lf %lf %lf\n", m_accEpochs, m_learningRate, m_assess.trainSSE, m_assess.trainRatio);

	m_pInput->saveWeights(wfile);
	m_pHidden0->saveWeights(wfile);
	if (HIDDEN_LAYER_NUMBER == 2)
	{
		m_pHidden1->saveWeights(wfile);
	}

	fclose(wfile);	
	return 0;
}

int NeuralNetwork::loadWeights(const char* fileName) {
	
	FILE* wfile = fopen(fileName, "r");
	if (wfile == nullptr) {
		printf("NeuralNetwork::loadWeights-file does not exist!");
		return -1;
	}

	int iN, h0N, h1N, oN;
	iN = h0N = h1N = oN = 0;
	fscanf(wfile, "%d %d ", &iN, &h0N);
	if (HIDDEN_LAYER_NUMBER == 2)
		fscanf(wfile, "%d \n", &h1N);
	fscanf(wfile, "%d ", &oN);
	int af = 0;
	fscanf(wfile, "%d\n", &af);

	if (af == 0)
		m_af = ActivationFunction::AF_RELU;
	else if (af == 1)
		m_af = ActivationFunction::AF_SIGMOID;
	else
		m_af = ActivationFunction::AF_TANH;
	setActivationFunction(m_af);

	fscanf(wfile, "%d %lf %lf %lf\n", &m_accEpochs, &m_learningRate, &m_assess.trainSSE, &m_assess.trainRatio);

	m_pInput->loadWeights(wfile);
	m_pHidden0->loadWeights(wfile);
	if (HIDDEN_LAYER_NUMBER == 2)
	{
		m_pHidden1->loadWeights(wfile);
	}

	fclose(wfile);
	return 0;
}

void NeuralNetwork::loadTrainPatterns(const char* fileName) {
	loadPatterns(fileName, m_train);
}

void NeuralNetwork::loadTestPatterns(const char* fileName) {
	loadPatterns(fileName, m_test);
}

void NeuralNetwork::loadPatterns(const char* fileName, CArray<Letter_S>& patternSet) {
	FILE* pfile = fopen(fileName, "r");
	if (pfile == nullptr) {
		printf("NeuralNetwork::loadPatterns could not open file %s!", fileName);
		return;
	}
	Letter_S letter;
	while (true) {
		letter.reset();
		char c; 
		if (fscanf(pfile, "%c", &c) == EOF)
			return;
		if (c < 'A' || c > 'Z')
			continue;

		letter.symbol = c;
		int f = 0;
		for (int i = 0; i < INPUT_NEURONS; i++)
		{
			if (fscanf(pfile, ",%d", &f) == EOF)
				return;
			else
				letter.X[i] = f/15.0; //normalize
		}
		letter.O[letter.symbol - 'A'] = 1;

		patternSet.Add(letter);
	}
}
Letter_S* NeuralNetwork::parsePattern(CString& pattern, Letter_S* letter)
{
	if (letter == nullptr || pattern.GetLength() == 0)
		return nullptr;
	pattern.Trim();
	letter->reset();
	
	CString sToken = _T("");
	for (int i = 0; i < INPUT_NEURONS; i++)
	{
		AfxExtractSubString(sToken, pattern, i, ',');
		letter->X[i] = ::StrToInt(sToken)/15.0;
	}
	return letter;
}

void NeuralNetwork::shuffleTrainData()
{
	int pattern_size = m_train.GetSize();
	for (int i=0; i<pattern_size; i++)
	{
		Letter_S letter = m_train.GetAt(i);
		int index = rand() % (pattern_size-i);
		m_train[i] = m_train[index];
		m_train[index] = letter;
	}
}

void NeuralNetwork::setInput(double* input)
{
	for (int i = 0; i < INPUT_NEURONS; i++)
		m_pInput->m_neurons[i]->m_o = input[i];
}

/*
 *  Feedforward the inputs of the neural network to the outputs.
 */
void NeuralNetwork::feedForward()
{
	m_pHidden0->calcOutput(m_pInput);
	if (HIDDEN_LAYER_NUMBER == 1) 
	{
		m_pOutput->calcOutput(m_pHidden0);
	}
	else
	{
		m_pHidden1->calcOutput(m_pHidden0);
		m_pOutput->calcOutput(m_pHidden1);
	}
}

void NeuralNetwork::setLearningRate(double rate)
{
	m_learningRate = rate;
}
/*
 *  Backpropagate the error through the network.
 */
void NeuralNetwork::backPropagate(void)
{
	m_pOutput->calcError(nullptr);
	if (HIDDEN_LAYER_NUMBER == 1)
	{
		m_pHidden0->updateWeights(m_pOutput, m_learningRate);
		m_pHidden0->calcError(m_pOutput);
	}
	else
	{
		m_pHidden1->updateWeights(m_pOutput, m_learningRate);
		m_pHidden1->calcError(m_pOutput);
		m_pHidden0->updateWeights(m_pHidden1, m_learningRate);
		m_pHidden0->calcError(m_pHidden1);
	}
	m_pInput->updateWeights(m_pHidden0, m_learningRate);
}

char NeuralNetwork::classify(double* vals) {
	int index = -1;
	double max = 0;
	for (int i = 0; i < OUTPUT_NEURONS; i++)
	{
		if (vals[i] > max)
		{
			max = vals[i];
			index = i;
		}
	}
	return 'A' + index;
}

Assess_S* NeuralNetwork::assess()
{
	memset(&m_assess, 0, sizeof(Assess_S));
	if (m_train.GetSize() == 0 && m_test.GetSize() == 0)
		return &m_assess;

	int correct = 0;
	//train
	int patternSize = m_train.GetSize();
	for (int i = 0; i < patternSize; i++)
	{
		Letter_S& pattern = m_train[i];
		setInput(pattern.X);
		feedForward();

		double* output = m_pOutput->getOutput();
		for (int k = 0; k < OUTPUT_NEURONS; k++) {
			m_assess.trainSSE += sqr(pattern.O[k] - output[k]);
		}
		if (classify(output) == pattern.symbol)
			correct++;
	}
	if (patternSize > 0)
	{
		m_assess.trainMse = m_assess.trainSSE / patternSize;
		m_assess.trainSSE /= 2;
		m_assess.trainRatio = correct * 1.0 / patternSize;
	}
	//test
	correct = 0;
	patternSize = m_test.GetSize();
	for (int i = 0; i < patternSize; i++)
	{
		Letter_S& pattern = m_test[i];
		setInput(pattern.X);
		feedForward();

		double* output = m_pOutput->getOutput();
		for (int k = 0; k < OUTPUT_NEURONS; k++) {
			m_assess.testSSE += sqr(pattern.O[k] - output[k]);
		}
		char predicted = classify(output);
		m_assess.confusionMatrix[pattern.symbol - 'A'][predicted - 'A'] += 1;
		if (predicted == pattern.symbol)
			correct++;
	}
	if (patternSize > 0)
	{
		m_assess.testMse = m_assess.testSSE / patternSize;
		m_assess.testSSE /= 2;
		m_assess.testRatio = correct * 1.0 / patternSize;
	}

	return &m_assess;
}

void NeuralNetwork::makeCache(CArray<double*>& cache, int number, int itemSize) {
	for (int i = 0; i < number; i++)
		cache.Add(new double[itemSize]);
}

void NeuralNetwork::clearCache(CArray<double*>& cache)
{
	for (int i = 0; i < cache.GetSize(); i++)
		delete cache[i];
	cache.RemoveAll();
}

double NeuralNetwork::preTrainNetwork(int epochsPretrain, BOOL lastLayer)
{
	const int number_of_pattern = m_train.GetSize();
	const int max_epochs = epochsPretrain;
	if (HIDDEN_LAYER_NUMBER == 1 || number_of_pattern == 0)
		return 0;

	double accumulatedErr = 0.0;
	int epochs = 0;
	int sample = 0;

	//m_pInput, m_pHidden0
	Autoencoder encoder0(m_pInput->m_N, m_pHidden0->m_N, m_pInput->m_N, m_learningRate);
	while (epochs < max_epochs) {
		if (sample == number_of_pattern) {
			sample = 0;
			accumulatedErr = 0;
			epochs++;
		}
		Letter_S& pattern = m_train[sample++];
		accumulatedErr += encoder0.train(pattern.X, pattern.X);
	}
	m_pInput->copyWeights(encoder0.getInputLayer());

	CArray<double*> cache; //16000*32
	makeCache(cache, number_of_pattern, m_pHidden0->m_N);
	int itemSize = sizeof(double)*m_pHidden0->m_N;
	for (int i = 0; i < number_of_pattern; i++) {
		setInput(m_train[i].X);
		m_pHidden0->calcOutput(m_pInput);
		memcpy(cache[i], m_pHidden0->getOutput(), itemSize);
	}
	//m_pHidden0, m_pHidden1
	accumulatedErr = 0.0;
	epochs = 0;
	sample = 0;
	Autoencoder encoder1(m_pHidden0->m_N, m_pHidden1->m_N, m_pHidden0->m_N, m_learningRate);
	while (epochs < max_epochs) {
		if (sample == number_of_pattern) {
			sample = 0;
			accumulatedErr = 0;
			epochs++;
		}
		accumulatedErr += encoder1.train(cache[sample], cache[sample]);
		sample++;
	}
	clearCache(cache);
	m_pHidden0->copyWeights(encoder1.getInputLayer());

	//last layer, m_pHidden1
	if (lastLayer) {
		//calculate the input and cache
		makeCache(cache, number_of_pattern, m_pHidden1->m_N);
		int itemSize = sizeof(double)*m_pHidden1->m_N;
		for (int i = 0; i < number_of_pattern; i++) {
			setInput(m_train[i].X);
			m_pHidden0->calcOutput(m_pInput);
			m_pHidden1->calcOutput(m_pHidden0);
			memcpy(cache[i], m_pHidden1->getOutput(), itemSize);
		}
		accumulatedErr = 0.0;
		epochs = 0;
		sample = 0;
		Autoencoder encoder2(m_pHidden1->m_N, m_pOutput->m_N, m_pOutput->m_N, m_learningRate);
		while (epochs < max_epochs*2) {
			if (sample == number_of_pattern) {
				sample = 0;
				accumulatedErr = 0;
				epochs++;
			}
			accumulatedErr += encoder1.train(cache[sample], m_train[sample].O);
			sample++;
		}
		clearCache(cache);
		m_pHidden1->copyWeights(encoder2.getInputLayer());
	}

	return accumulatedErr;
}

double* NeuralNetwork::test(Letter_S* testPattern) {
	setInput(testPattern->X);
	feedForward();
	memcpy(testPattern->O, m_pOutput->getOutput(), sizeof(double)*OUTPUT_NEURONS);
	return testPattern->O;
}

int NeuralNetwork::train(int numberOfEpochs, double learningRate, 
	BOOL pretrain, int epochsPretrain, BOOL lastLayer)
{
	int number_of_pattern = m_train.GetSize();
	if (number_of_pattern == 0) {
		return -1;
	}

	if (pretrain && !m_preTrained)
	{
		preTrainNetwork(epochsPretrain, lastLayer);
		m_preTrained = TRUE;
	}
	m_accEpochs += numberOfEpochs;
	m_learningRate = learningRate;

	double accumulatedErr = 0.0;
	int epochs = 0;
	double m_sse = 0.0;
	int sample = 0;
	while (epochs < numberOfEpochs) {
		if (sample == number_of_pattern) {
			sample = 0;
			m_sse = 0.0;
			accumulatedErr = 0;
			epochs++;
		}
		
		Letter_S& pattern = m_train[sample++];
		setInput(pattern.X);
		m_pOutput->setDesiredOuput(pattern.O);
		feedForward();
		backPropagate();

		double* output = m_pOutput->getOutput();
		for (int k = 0; k < OUTPUT_NEURONS; k++) {
			m_sse += sqr(pattern.O[k] - output[k]);
		}

		m_sse = 0.5 * m_sse;
		accumulatedErr += m_sse;
	}

	return 0;
}
