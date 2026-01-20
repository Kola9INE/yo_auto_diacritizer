import streamlit as st
import torch
import torch.nn as nn
import joblib
from pathlib import Path

path = Path(__file__).parent
class BiLSTMDiacritizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, embedding_weights=None):
        super(BiLSTMDiacritizer, self).__init__()
        
        # 1. Embedding Layer
        # If weights are provided, use them. Otherwise, initialize randomly.
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        
        # 3. Fully Connected Layer
        # Input size is hidden_dim * 2 because it's bidirectional (Forward + Backward)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # Pass through embedding
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch size, seq len, embedding dim]
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs shape: [batch size, seq len, hidden_dim * 2]
        
        # Pass through Linear layer to get predictions
        predictions = self.fc(self.dropout(outputs))
        # predictions shape: [batch size, seq len, output_dim]
        
        return predictions

input_word2idx = joblib.load(Path(path/'PHASE_TWO'/'input_word2idx.pkl'))
target_word2idx = joblib.load(Path(path/'PHASE_TWO'/'target_word2idx.pkl'))
target_idx2word = joblib.load(Path(path/'PHASE_TWO'/'target_idx2word.pkl'))
embedding_matrix = joblib.load(Path(path/'PHASE_TWO'/'yoruba_embedding_matrix.joblib'))
embedding_weights = torch.tensor(
    embedding_matrix,
    dtype=torch.float32
)

# --- Configuration ---
INPUT_DIM = len(input_word2idx)
OUTPUT_DIM = len(target_word2idx)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3
PAD_IDX = 0

def predict_sentence(sentence, model, device):
    """
    Takes a raw sentence string (undiacritized) and returns the diacritized version.
    """
    model.eval()
    
    # Tokenize (Simple split)
    tokens = str(sentence).split()
    
    # Numericalize (Handle Unknowns)
    token_ids = [input_word2idx.get(t, input_word2idx.get("<UNK>", 1)) for t in tokens]
    
    # Convert to Tensor
    tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device) # Shape: [1, seq_len]
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        # output shape: [1, seq_len, output_dim]
        
        # Get max probability indices
        prediction_indices = output.argmax(dim=2).squeeze(0).tolist()
    
    # Convert indices back to words
    predicted_words = [target_idx2word.get(idx, "<UNK>") for idx in prediction_indices]
    
    return " ".join(predicted_words)

@st.cache_resource
def load_model():
        model = BiLSTMDiacritizer(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX, embedding_weights=embedding_weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(Path(path/'PHASE_TWO'/'best-model.pt'), map_location=device))
        model = model.to(device)
        return model, device

if __name__ == '__main__':
    model, device = load_model()
    st.set_page_config(page_title="Yorùbá Text Diacritization", layout="wide")
    st.title("Yorùbá Text Diacritization")
    text = st.text_area("Enter random text here")
    col1, col2 = st.columns(2)
    if st.button("Diacritize"):
        if not text or text == '':
            st.snow()
            st.warning('PLEASE INPUT ANY TEXT IN THE TEXT AREA ABOVE')
        
        else:
            diacritized_text = predict_sentence(text, model, device)
            col1.subheader('Diacritized Text:')
            col1.code(diacritized_text)