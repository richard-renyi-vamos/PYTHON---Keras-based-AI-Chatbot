import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import scrolledtext

# Sample conversation data
conversations = [
    ("Hello", "Hi there!"),
    ("How are you?", "I'm doing well, thanks for asking."),
    ("What's your name?", "I'm an AI chatbot."),
    ("Goodbye", "See you later!")
]

# Prepare the data
inputs = [conv[0] for conv in conversations]
outputs = [conv[1] for conv in conversations]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + outputs)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)

# Pad sequences
max_sequence_length = max(len(seq) for seq in input_sequences + output_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# Create the model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_sequence_length),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, np.array(output_sequences).reshape(len(output_sequences), max_sequence_length, 1), epochs=100)

# Function to generate a response
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    predicted_seq = model.predict(input_seq)[0]
    predicted_words = []
    for i in predicted_seq:
        predicted_word = tokenizer.index_word[np.argmax(i)]
        if predicted_word == '<PAD>':
            break
        predicted_words.append(predicted_word)
    return ' '.join(predicted_words)

# Create the main window
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("400x500")

# Create and pack the chat display
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
chat_display.pack(padx=10, pady=10)

# Create and pack the input field
input_field = tk.Entry(root, width=40)
input_field.pack(side=tk.LEFT, padx=10)

# Function to handle sending a message
def send_message():
    user_input = input_field.get()
    if user_input:
        chat_display.insert(tk.END, f"You: {user_input}\n")
        response = generate_response(user_input)
        chat_display.insert(tk.END, f"Bot: {response}\n\n")
        input_field.delete(0, tk.END)
    chat_display.see(tk.END)

# Create and pack the send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=10)

# Bind the Enter key to send_message function
root.bind('<Return>', lambda event: send_message())

# Start the GUI event loop
root.mainloop()
