
// LetterRecognitionDlg.cpp : implementation file
//

#include "stdafx.h"
#include "LetterRecognition.h"
#include "LetterRecognitionDlg.h"
#include "afxdialogex.h"
#include "backpropagation.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()
// CLetterRecognitionDlg dialog

UINT LETTER_IDS[OUTPUT_NEURONS+24] = { IDC_A, IDC_B, IDC_C, IDC_D, IDC_E, IDC_F, IDC_G, IDC_H, IDC_I,
			IDC_J, IDC_K, IDC_L, IDC_M, IDC_N, IDC_O, IDC_P, IDC_Q, IDC_R,
			IDC_S, IDC_T,  IDC_U, IDC_V, IDC_W, IDC_X, IDC_Y, IDC_Z };

CLetterRecognitionDlg::CLetterRecognitionDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_LETTERRECOGNITION_DIALOG, pParent)
	, m_iMaxEpochs(DEFAULT_MAX_EPOCHS)
	, m_dTestSSE(0)
	, m_dTrainSSE(0)
	, m_dTrainPercent(0)
	, m_dLearningRate(RELU_LEARNING_RATE)
	, m_dTestPercent(0)
	, m_sTest(_T(""))
	, m_sTestFile(_T(""))
	, m_sTrainFile(_T(""))
	, m_epochsPretrain(5)
	, m_loadWFile(_T(""))
	, m_saveWFile(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CLetterRecognitionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_MAX_EPOCHS, m_iMaxEpochs);
	DDX_Text(pDX, IDC_TEST_SSE, m_dTestSSE);
	DDX_Text(pDX, IDC_TRAIN_SSE, m_dTrainSSE);
	DDX_Text(pDX, IDC_TRAIN_GOOD_PERCENT, m_dTrainPercent);
	DDX_Text(pDX, IDC_LEARNING_RATE, m_dLearningRate);
	DDX_Text(pDX, IDC_TEST_GOOD_PERCENT, m_dTestPercent);
	DDX_Text(pDX, IDC_TEST_INPUT, m_sTest);
	DDX_Text(pDX, IDC_TEST_FILE, m_sTestFile);
	DDX_Text(pDX, IDC_TRAIN_FILE, m_sTrainFile);
	DDX_Control(pDX, IDC_SPIN_EPOCHS, m_spinEpochs);
	DDX_Control(pDX, IDC_SPIN_RATE, m_spinRate);
	DDX_Control(pDX, IDC_MAX_EPOCHS, m_editEpochs);
	DDX_Control(pDX, IDC_LEARNING_RATE, m_editRate);
	DDX_Control(pDX, IDC_CHECK_PRETRAIN, m_chkPretrain);
	DDX_Control(pDX, IDC_PRETRAIN_EPOCHS, m_editPretrain);
	DDX_Text(pDX, IDC_PRETRAIN_EPOCHS, m_epochsPretrain);
	DDX_Control(pDX, IDC_CHECK_LAST_LAYER, m_chkLastLayer);
	DDX_Text(pDX, IDC_LOAD_WEIGHTS, m_loadWFile);
	DDX_Text(pDX, IDC_SAVE_WEIGHTS, m_saveWFile);
	DDX_Control(pDX, IDC_COMBO_ACTIVATION_FUNCTION, m_comboFunc);
	DDX_Control(pDX, IDC_PRE_TRAIN, m_btnPreTrain);
}

BEGIN_MESSAGE_MAP(CLetterRecognitionDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_TRAIN_FILE, &CLetterRecognitionDlg::OnBnClickedBtnTrainFile)
	ON_BN_CLICKED(IDC_BTN_TEST, &CLetterRecognitionDlg::OnBnClickedBtnTest)
	ON_BN_CLICKED(IDC_BTN_TRAIN_NN, &CLetterRecognitionDlg::OnBnClickedBtnTrainNn)
	ON_NOTIFY(UDN_DELTAPOS, IDC_SPIN_EPOCHS, &CLetterRecognitionDlg::OnDeltaposSpinEpochs)
	ON_NOTIFY(UDN_DELTAPOS, IDC_SPIN_RATE, &CLetterRecognitionDlg::OnDeltaposSpinRate)
	ON_BN_CLICKED(IDC_BTN_TEST_FILE, &CLetterRecognitionDlg::OnBnClickedBtnTestFile)
	ON_BN_CLICKED(IDC_CHECK_PRETRAIN, &CLetterRecognitionDlg::OnBnClickedCheckPretrain)
	ON_BN_CLICKED(IDC_BTN_SAVE_WEIGHTS, &CLetterRecognitionDlg::OnBnClickedBtnSaveWeights)
	ON_BN_CLICKED(IDC_BTN_LOAD_WEIGHTS, &CLetterRecognitionDlg::OnBnClickedBtnLoadWeights)
	ON_BN_CLICKED(IDC_BTN_SHUFFLE, &CLetterRecognitionDlg::OnBnClickedBtnShuffle)
	ON_BN_CLICKED(IDC_BTN_ASSESS, &CLetterRecognitionDlg::OnBnClickedBtnAssess)
	ON_CBN_SELCHANGE(IDC_COMBO_ACTIVATION_FUNCTION, &CLetterRecognitionDlg::OnCbnSelchangeComboActivationFunction)
	ON_BN_CLICKED(IDC_PRE_TRAIN, &CLetterRecognitionDlg::OnBnClickedPreTrain)
END_MESSAGE_MAP()

// CLetterRecognitionDlg message handlers

BOOL CLetterRecognitionDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	m_spinEpochs.SetBuddy(&m_editEpochs);
	m_spinEpochs.SetRange(1, 100);
	m_spinEpochs.SetPos(2);
	m_spinRate.SetBuddy(&m_editRate);
	m_spinRate.SetRange(1, 1000);
	m_spinRate.SetPos(10);

	m_editPretrain.EnableWindow(FALSE);
	m_chkLastLayer.SetCheck(FALSE);
	m_chkLastLayer.EnableWindow(FALSE);
	m_btnPreTrain.EnableWindow(FALSE);

	m_comboFunc.AddString(_T("ReLU"));
	m_comboFunc.SetItemData(0, ActivationFunction::AF_RELU);
	m_comboFunc.AddString(_T("Sigmoid"));
	m_comboFunc.SetItemData(1, ActivationFunction::AF_SIGMOID);
	m_comboFunc.AddString(_T("Tanh"));
	m_comboFunc.SetItemData(2, ActivationFunction::AF_TANH);
	m_comboFunc.SetCurSel(0);

	m_sTest = _T("6,9,8,4,3,8,7,3,4,13,5,8,6,8,0,8");
	GetDlgItem(IDC_TEST_INPUT)->SetWindowTextA(m_sTest);

	m_nn.initialise();

	return TRUE; 
}

void CLetterRecognitionDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

void CLetterRecognitionDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CLetterRecognitionDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CLetterRecognitionDlg::OnDeltaposSpinEpochs(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
	if ((pNMUpDown->iDelta < 0) && (m_iMaxEpochs > 5))
		m_iMaxEpochs -= 5;
	if ((pNMUpDown->iDelta > 0) && (m_iMaxEpochs < 495))
		m_iMaxEpochs += 5;
	UpdateData(false);
	*pResult = 0;
}

void CLetterRecognitionDlg::OnDeltaposSpinRate(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
	if ((pNMUpDown->iDelta < 0) && (m_dLearningRate >= 0.02))
		m_dLearningRate -= 0.01;
	if ((pNMUpDown->iDelta > 0) && (m_dLearningRate <= 9.99))
		m_dLearningRate += 0.01;
	UpdateData(false);
	*pResult = 0;
}

void CLetterRecognitionDlg::OnCbnSelchangeComboActivationFunction()
{
	int index = m_comboFunc.GetCurSel();
	if (index == 0)
		m_dLearningRate = RELU_LEARNING_RATE;
	else if (index == 1)
		m_dLearningRate = SIGMOID_LEARNING_RATE;
	else
		m_dLearningRate = TANH_LEARNING_RATE;

	m_nn.setActivationFunction((ActivationFunction)m_comboFunc.GetItemData(index));
	UpdateData(FALSE);
}

void CLetterRecognitionDlg::OnBnClickedBtnTrainFile()
{
	CFileDialog dlg(TRUE, CString(".txt"), NULL, OFN_READONLY, CString("Text file(*.txt)|*.txt|All files (*.*)|*.*||"));
	if (dlg.DoModal() == IDOK)
	{
		m_sTrainFile = dlg.GetPathName();
		m_nn.loadTrainPatterns(m_sTrainFile.GetBuffer(m_sTrainFile.GetLength() + 1));
		UpdateData(FALSE);
	}
}

void CLetterRecognitionDlg::OnBnClickedCheckPretrain()
{
	if (m_chkPretrain.GetCheck() == BST_CHECKED) {
		m_editPretrain.EnableWindow(TRUE);
		m_chkLastLayer.EnableWindow(TRUE);
		m_btnPreTrain.EnableWindow(TRUE);
	}
	else {
		m_editPretrain.EnableWindow(FALSE);
		m_chkLastLayer.EnableWindow(FALSE);
		m_btnPreTrain.EnableWindow(FALSE);
	}
}

void CLetterRecognitionDlg::OnBnClickedBtnTestFile()
{
	CFileDialog dlg(TRUE, CString(".txt"), NULL, OFN_READONLY, CString("Text file(*.txt)|*.txt|All files (*.*)|*.*||"));
	if (dlg.DoModal() == IDOK)
	{
		m_sTestFile = dlg.GetPathName();
		m_nn.loadTestPatterns(m_sTestFile.GetBuffer(m_sTestFile.GetLength() + 1));
		UpdateData(FALSE);
	}
}

void CLetterRecognitionDlg::OnBnClickedBtnSaveWeights()
{
	CFileDialog dlg(FALSE, CString(".txt"), NULL, OFN_OVERWRITEPROMPT, CString("Text file(*.txt)|*.txt|All files (*.*)|*.*||"));
	if (dlg.DoModal() == IDOK)
	{
		m_saveWFile = dlg.GetPathName();
		m_nn.saveWeights(m_saveWFile.GetBuffer(m_saveWFile.GetLength() + 1));
		UpdateData(FALSE);
	}
}

void CLetterRecognitionDlg::OnBnClickedBtnLoadWeights()
{
	CFileDialog dlg(TRUE, CString(".txt"), NULL, OFN_READONLY, CString("Text file(*.txt)|*.txt|All files (*.*)|*.*||"));
	if (dlg.DoModal() == IDOK)
	{
		m_loadWFile = dlg.GetPathName();
		m_nn.loadWeights(m_loadWFile.GetBuffer(m_loadWFile.GetLength() + 1));
		m_comboFunc.SetCurSel(m_nn.getActivationFunction());
		OnCbnSelchangeComboActivationFunction();
		UpdateData(FALSE);
	}
}

void CLetterRecognitionDlg::OnBnClickedBtnTest()
{
	UpdateData();
	Letter_S letter;
	Letter_S* pLetter = m_nn.parsePattern(m_sTest, &letter);
	if (pLetter == nullptr) {
		AfxMessageBox(_T("There is error in the pattern, please correct and retry."), MB_OK | MB_ICONSTOP);
		return;
	}

	m_nn.test(pLetter);

	char str[10];
	for (int i = 0; i < OUTPUT_NEURONS; i++) {
		sprintf(str, "%.7f", pLetter->O[i]);
		GetDlgItem(LETTER_IDS[i])->SetWindowTextA(_T(str));
	}
	memset(str, 0, 10);
	char c = m_nn.classify(pLetter->O);
	sprintf(str, "%c", c);
	GetDlgItem(IDC_TEST_RESULT)->SetWindowTextA(_T(str));

	UpdateData(FALSE);
}

void CLetterRecognitionDlg::OnBnClickedBtnTrainNn()
{
	UpdateData();
	BOOL pretrain = (m_chkPretrain.GetCheck() == BST_CHECKED);
	BOOL lastLayer = (m_chkLastLayer.GetCheck() == BST_CHECKED);
	//no pretrain as we can do it separately from UI
	if (-1 == m_nn.train(m_iMaxEpochs, m_dLearningRate, FALSE, m_epochsPretrain, lastLayer))
		AfxMessageBox(_T("No training patterns loaded.."), MB_OK | MB_ICONSTOP);

	OnBnClickedBtnAssess();
}

void CLetterRecognitionDlg::OnBnClickedButton1()
{
	Backpropagation bp;
	bp.loadPatterns("C:\\AI\\NN\\Training_Data.txt");
	bp.trainNetwork(100);

	double* result;
	char str[10];

	m_nn.loadTestPatterns("C:\\AI\\NN\\Test_Data.txt");
	int correct = 0;
	for (int i = 0; i < m_nn.m_test.GetSize(); i++) {
		result = bp.testNetwork(&m_nn.m_test[i]);
		char c = m_nn.classify(result);
		if (c == m_nn.m_test[i].symbol)
			correct++;
	}
	double rate = correct * 1.0 / m_nn.m_test.GetSize();
	memset(str, 0, 10);
	sprintf(str, "%.2f", rate);

	//train
	correct = 0;
	for (int i = 0; i < bp.m_train.GetSize(); i++) {
		result = bp.testNetwork(&bp.m_train[i]);
		char c = m_nn.classify(result);
		if (c == bp.m_train[i].symbol)
			correct++;
	}
	rate = correct * 1.0 / bp.m_train.GetSize();
	memset(str, 0, 10);
	sprintf(str, "%.2f", rate);
}

void CLetterRecognitionDlg::OnBnClickedBtnShuffle()
{
	m_nn.shuffleTrainData();
}

void CLetterRecognitionDlg::OnBnClickedBtnAssess()
{
	Assess_S* assess = m_nn.assess();
	char str[10];
	memset(str, 0, 10);
	sprintf(str, "%.2f", assess->trainSSE);
	GetDlgItem(IDC_TRAIN_SSE)->SetWindowTextA(_T(str));
	memset(str, 0, 10);
	sprintf(str, "%.2f", assess->trainRatio * 100);
	GetDlgItem(IDC_TRAIN_GOOD_PERCENT)->SetWindowTextA(_T(str));
	memset(str, 0, 10);
	sprintf(str, "%.2f", assess->testSSE);
	GetDlgItem(IDC_TEST_SSE)->SetWindowTextA(_T(str));
	memset(str, 0, 10);
	sprintf(str, "%.2f", assess->testRatio * 100);
	GetDlgItem(IDC_TEST_GOOD_PERCENT)->SetWindowTextA(_T(str));
	if (IDNO == AfxMessageBox(_T("Do you want to save the Confusion Marix for testing data?"), MB_YESNO | MB_ICONQUESTION))
		return;

	//save confusion matrix
	CFileDialog dlg(FALSE, CString(".txt"), NULL, OFN_OVERWRITEPROMPT, CString("Text file(*.txt)|*.txt|All files (*.*)|*.*||"));
	if (dlg.DoModal() == IDOK)
	{
		CString cmFile = dlg.GetPathName();		
		FILE* wfile = fopen(cmFile.GetBuffer(cmFile.GetLength() + 1), "w");
		//save network structure
		for (int i = 0; i < OUTPUT_NEURONS; i++)
			fprintf(wfile, "\t%c", 'A' + i);
		fprintf(wfile, "\n");
		for (int i = 0; i < OUTPUT_NEURONS; i++)
		{
			fprintf(wfile, "%c\t", 'A' + i);
			for (int j = 0; j < OUTPUT_NEURONS; j++)
			{
				fprintf(wfile, "%d\t", assess->confusionMatrix[i][j]);
			}
			fprintf(wfile, "\n");
		}
		fclose(wfile);
	}
}

void CLetterRecognitionDlg::OnBnClickedPreTrain()
{
	UpdateData();
	m_nn.setLearningRate(m_dLearningRate);
	BOOL lastLayer = (m_chkLastLayer.GetCheck() == BST_CHECKED);
	m_nn.preTrainNetwork(m_epochsPretrain, lastLayer);
	AfxMessageBox(_T("Pre-train completed!"), MB_OK | MB_ICONINFORMATION);
}
