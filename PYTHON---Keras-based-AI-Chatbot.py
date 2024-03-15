import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Define a simple dataset of input-output pairs
training_data = np.array([
    ("Hello", "Hi there!"),
    ("How are you?", "I'm fine, thank you!"),
    ("What is your name?", "I'm a chatbot."),
    ("Goodbye", "See you later!"),
])

# Create a simple model
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model using the input-output pairs
model.fit(training_data[:,0], training_data[:,1], epochs=100, batch_size=1)

# Function to get the model's response to input text
def get_response(input_text):
    input_vector = np.array([hash(input_text)])  # Convert text to a numerical representation
    prediction = model.predict(input_vector)[0][0]
    return training_data[np.argmin(np.abs(prediction - np.array(list(map(hash, training_data[:,0])))))][1]

# Create GUI
def send_message(event):
    input_text = entry.get()
    response = get_response(input_text)
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "You: " + input_text + "\n")
    chat_box.insert(tk.END, "Chatbot: " + response + "\n")
    chat_box.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

# Create main window
root = tk.Tk()
root.title("Chatbot")

# Create chat display
chat_box = tk.Text(root, height=20, width=50)
chat_box.config(state=tk.DISABLED)
chat_box.pack()

# Create input box
entry = tk.Entry(root, width=50)
entry.bind("<Return>", send_message)
entry.pack()

root.mainloop()
