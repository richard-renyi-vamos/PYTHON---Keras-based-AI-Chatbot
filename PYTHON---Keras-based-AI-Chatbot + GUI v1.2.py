import tkinter as tk
from tkinter import filedialog, Text, messagebox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Function to create and compile the model
def create_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to open file dialog and return selected file paths
def load_files():
    file_paths = filedialog.askopenfilenames(title="Select Text Documents", filetypes=[("Text Files", "*.txt")])
    return list(file_paths)

# Function to preprocess text data
def preprocess_text(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return padded_sequences, tokenizer.word_index, max_length

# Function to train the model
def train_model(file_paths):
    texts, word_index, max_length = preprocess_text(file_paths)
    vocab_size = len(word_index) + 1
    embedding_dim = 16

    model = create_model(vocab_size, embedding_dim, max_length)
    
    # For simplicity, we'll use the same texts as both features and labels
    labels = np.array([1] * len(texts))
    
    model.fit(texts, labels, epochs=10, verbose=1)
    return model

# Function to display results in the GUI
def display_results(model):
    result_window = tk.Toplevel()
    result_window.title("Training Results")

    result_label = tk.Label(result_window, text=f"Model Summary:\n{model.summary()}")
    result_label.pack()

    close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
    close_button.pack()

# Function to handle chatting with the model
def chat_with_model(model, tokenizer, max_length):
    def send_message():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)
        try:
            detected_lang = detect(user_input)
        except LangDetectException:
            detected_lang = "unknown"
        
        if detected_lang != "en":
            messagebox.showinfo("Language Detection", f"Detected language: {detected_lang}\nCurrently, only English is supported.")
            return
        
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        prediction = model.predict(padded_sequence)[0][0]
        response = "Positive" if prediction > 0.5 else "Negative"
        
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You: {user_input}\n")
        chat_log.insert(tk.END, f"Model: {response}\n")
        chat_log.config(state=tk.DISABLED)

    chat_window = tk.Toplevel()
    chat_window.title("Chat with the Model")

    chat_log = tk.Text(chat_window, state=tk.DISABLED, width=50, height=20)
    chat_log.pack()

    user_entry = tk.Entry(chat_window, width=50)
    user_entry.pack()
    
    send_button = tk.Button(chat_window, text="Send", command=send_message)
    send_button.pack()

def main():
    # Main window setup
    root = tk.Tk()
    root.title("Neural Network Text Trainer")

    # Select Files Button
    select_files_button = tk.Button(root, text="Select Text Documents", padx=10, pady=5, fg="white", bg="blue", command=load_files)
    select_files_button.pack()

    # Language Selection
    language_label = tk.Label(root, text="Select Language:")
    language_label.pack()
    language_var = tk.StringVar(value="English")
    language_menu = tk.OptionMenu(root, language_var, "English", "Spanish", "French", "German")
    language_menu.pack()

    # Train Button
    def on_train():
        file_paths = load_files()
        if not file_paths:
            messagebox.showwarning("No files selected", "Please select text documents to train the model.")
            return
        
        model = train_model(file_paths)
        display_results(model)
        chat_button.config(state=tk.NORMAL)
        root.model = model  # Store the model in the root window
        root.tokenizer, root.max_length = preprocess_text(file_paths)[1:]

    train_button = tk.Button(root, text="Train Model", padx=10, pady=5, fg="white", bg="green", command=on_train)
    train_button.pack()

    # Chat Button
    chat_button = tk.Button(root, text="Chat with Model", padx=10, pady=5, fg="white", bg="purple", state=tk.DISABLED, command=lambda: chat_with_model(root.model, root.tokenizer, root.max_length))
    chat_button.pack()

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
