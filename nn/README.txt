Unzip the file, you may find:
    Assignment 2 Report.docx  -- report for assignment 2
    Letter_Feature_Extractor.html -- letter feature extractor, exported from Letter_Feature_Extractor.ipynb for quick reivew
    Letter_Feature_Extractor.ipynb -- please install WinPython or Anaconda2/3 to run, the code depends on opencv, install instruction is in the file
    letter_imgs  -- folder, contains testing image for Letter_Feature_Extractor.ipynb
    README.txt -- this file
    TestingResults -- Neural network testing results, initial/trained weights, confusion matrix and PGC-SSE-Epochs spread sheet
    VisualC++2017_Project -- folder, open "LetterRecognition.sln" and press F7 to build and F5 to run, you need Visual C++ 2017

Disable warnings:
1. _CRT_NONSTDC_NO_WARNINGS
   Configuration Properties >> C/C++ >> Preprocessor >>Preprocessor Definitions >> _CRT_NONSTDC_NO_WARNINGS
or 
   #pragma warning(disable:4996)