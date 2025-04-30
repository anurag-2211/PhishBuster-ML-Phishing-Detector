# PhishBuster - A Machine Learning Model to Stop Malicious Websites

PhishBuster is a cybersecurity-focused machine learning project that detects phishing websites based on various URL and domain-level features. It uses Random Forest and Support Vector Machine (SVM) classifiers to analyze and classify websites as either legitimate or phishing.

## About the Project

Phishing attacks are one of the most common online threats where users are tricked into revealing sensitive data through fake websites. PhishBuster aims to detect such malicious websites using machine learning techniques by analyzing patterns in their URLs and domain features.

## Features

- Uses two ML algorithms: Random Forest and SVM
- Feature scaling for better accuracy
- Detects phishing based on website structure and metadata
- Outputs predicted labels for new/unseen websites
- Displays predictions along with actual labels and URLs

## Dataset

- Format: CSV with 80+ features
- Contains columns:
  - `Website_Name`
  - `URL`
  - `Result` (1 = Legitimate, -1 = Phishing)

The dataset was cleaned and enhanced to include URL and website name for better readability in predictions.

## Algorithms Used

- Random Forest Classifier
- Support Vector Machine (SVM)

Both models are trained and evaluated using standard classification metrics like precision, recall, F1-score, and accuracy.

## How to Run

```bash
pip install pandas scikit-learn
python phishbuster_main.py
