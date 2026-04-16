# Fake News Detector

🛡️ A Machine Learning project to detect fake news using TF-IDF Vectorization and Naive Bayes Classification.

## Features
- **Real-time Analysis**: Enter article text to get a probability-based verdict.
- **Image Scanning (OCR)**: Use EasyOCR to extract and analyze text from images (e.g., screenshots of articles).
- **Performance Metrics**: View confusion matrices, precision, recall, and F1 scores.
- **Feature Insights**: Visualize which words are most predictive of Real vs. Fake news.

## Installation
Ensure you have Python installed, then install the dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib easyocr Pillow
```

## Dataset
This project uses the **WELFake Dataset** (approx. 72,000 news articles).
- Access it on Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- Place `WELFake_Dataset.csv` in the root directory to train the model.

## Usage
Run the main script:
```bash
python ML.py
```

## Technology Stack
- **Language**: Python
- **GUI**: Tkinter (Custom Dark Theme)
- **ML**: Scikit-Learn (Multinomial Naive Bayes, TfidfVectorizer)
- **OCR**: EasyOCR
- **Data**: Pandas, Numpy
- **Visualization**: Matplotlib
