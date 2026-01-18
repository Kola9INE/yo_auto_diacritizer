# Automatic Yoruba Diacritizer

This repository contains a machine learning project designed to automatically restore diacritics (tone marks) to Yoruba text. The system takes undiacritized Yoruba text as input and outputs the correctly diacritized version using a Bi-Directional LSTM (BiLSTM) model.

## 📂 Project Structure

### Data & Preprocessing

* **`pre_build.ipynb`**: The data collection notebook. It uses **Selenium** to scrape Yoruba articles from [Global Voices (Yoruba)](https://yo.globalvoices.org/). It handles dynamic content, extracts article bodies, and compiles them into raw text files.
* **`data.parquet`**: The processed dataset stored in a highly efficient binary format. It contains paired examples for training:
* `feature`: The input text (undiacritized).
* `label`: The target text (diacritized).


* **`yo_corpus.txt`, `news_sites.txt`, `owe.txt**`: Raw text files containing Yoruba corpora, proverbs, and news content used as source material for the dataset.
* **`char2idx.pkl`**: A pickle file containing the character-to-index mapping, essential for encoding text for the character-level model.

### Models & Training

* **`model_build.ipynb`**: The primary notebook for building and training the character-level BiLSTM diacritization model.
* **`fasttext-project.ipynb`**: An alternative experimental notebook that utilizes **FastText** embeddings combined with PyTorch to train the model, exploring word/subword level representations.
* **`char_bilstm_diacritizer.pt`**: The saved PyTorch model checkpoint (weights) for the Character BiLSTM Diacritizer.

### Utilities

* **`cln-720.ipynb`**: A utility notebook for data cleaning and exploration (likely used for specific corpus cleaning tasks).

## 🛠 Installation & Requirements

To run the notebooks and use the model, you will need **Python 3.x** and the following libraries:

```bash
pip install torch pandas numpy selenium fasttext scikit-learn tqdm pyarrow

```

*Note: For `pre_build.ipynb`, you will also need a compatible WebDriver (e.g., ChromeDriver) installed for Selenium.*

## 🚀 Usage

### 1. Data Collection (Optional)

If you wish to expand the dataset or scrape fresh data:

1. Open `pre_build.ipynb`.
2. Ensure you have `selenium` and the appropriate WebDriver configured.
3. Run the cells to scrape articles and generate the raw text corpus.

### 2. Model Training

To retrain the model or experiment with the architecture:

1. Open `model_build.ipynb` (for the Char-BiLSTM approach) or `fasttext-project.ipynb` (for the FastText approach).
2. Ensure `data.parquet` is in the correct directory.
3. Run the notebook cells to preprocess data, train the model, and evaluate performance.

### 3. Inference

You can use the pre-trained `char_bilstm_diacritizer.pt` model to diacritize new text. (See `model_build.ipynb` for the specific inference logic).

1. Load the model architecture and weights using `torch.load`.
2. Load the `char2idx.pkl` mapping.
3. Pass your undiacritized string into the model to generate the diacritized output.

## 📊 Dataset Details

The dataset is constructed primarily from the **Global Voices Yoruba** domain.

* **Input (Feature)**: Text stripped of tone marks (e.g., "ba wo ni").
* **Output (Label)**: Correctly toned text (e.g., "báwo ni").

## 🤝 Contributing

Contributions are welcome! If you have additional Yoruba corpora or improvements to the model architecture, please feel free to fork the repository and submit a pull request.

## 📄 License

[Apache 2.0]