## Next Word Prediction Using Deep Learning

---

**Project By:** Alaissa Shaikh
**Data Scientist**

---

**Introduction**

This project explores the application of deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, to tackle the challenging task of next word prediction. By modeling the complex relationships between words in a text sequence, the LSTM network learns to predict the most likely next word given a sequence of preceding words. This project demonstrates the potential of deep learning for natural language processing (NLP) tasks and paves the way for innovative language-based applications like autocompletion, chatbots, and creative writing assistance.

---

**Libraries**

The project utilizes the following Python libraries:

* `numpy (np)`: Numerical computing library for array manipulation.
* `tensorflow (tf)`: Deep learning framework for building and training the model.
* `tensorflow.keras.preprocessing.text.Tokenizer`: Tokenizes text data into numerical sequences.
* `tensorflow.keras.preprocessing.sequence.pad_sequences`: Pads sequences to a consistent length.
* `tensorflow.keras.models.Sequential`: Creates a sequential neural network model.
* `tensorflow.keras.layers.Embedding`: Embeds words into dense vector representations.
* `tensorflow.keras.layers.LSTM`: Captures long-term dependencies in sequential data.
* `tensorflow.keras.layers.Dense`: Outputs a probability distribution over the vocabulary.

---

**Data Preprocessing**

1. **Data Loading:**
   - Reads the text data from a file (e.g., `sherlock-holm.es_stories_plain-text_advs.txt`).

2. **Tokenization:**
   - Creates a `Tokenizer` object to map words to unique integer indices.
   - Fits the tokenizer on the entire text corpus.
   - Calculates the total number of words (`total_words`) in the vocabulary.

3. **Input-Output Sequence Creation:**
   - Iterates through each line of text.
   - Converts each line into a sequence of integer tokens using the tokenizer.
   - Creates input-output sequences by considering subsequences of tokens (n-grams).
   - The input sequence comprises all tokens except the last one, while the output (label) is the last token in the sequence.

4. **Padding:**
   - Determines the maximum sequence length (`max_sequence_len`) from all input sequences.
   - Pads shorter sequences with zeros at the beginning using `pad_sequences` to ensure consistent input shape for the model.

5. **Label Encoding:**
   - Converts integer labels (representing words) into one-hot encoded vectors using `tf.keras.utils.to_categorical`.
   - One-hot encoding facilitates the model's learning process by representing each label as a probability distribution over the vocabulary.

---

**Model Architecture**

A sequential deep learning model is constructed using `tensorflow.keras.models.Sequential`:

1. **Embedding Layer:**
   - Takes integer-encoded sequences as input.
   - Embeds each word into a dense vector of a specified dimensionality (e.g., 100 dimensions).
   - This layer captures semantic relationships between words.

2. **LSTM Layer:**
   - Processes the embedded sequences.
   - LSTMs are well-suited for sequential data as they can learn long-term dependencies between words in a sequence.
   - The number of LSTM units (e.g., 150) determines the model's capacity to capture these dependencies.

3. **Dense Layer:**
   - The final output layer.
   - Takes the output from the LSTM layer.
   - Generates a probability distribution over the vocabulary for each word in the sequence.
   - The word with the highest probability is predicted as the next word.

---

**Model Training**

1. **Compilation:**
   - Configures the model for training using `model.compile()`.
   - Specifies the loss function (`categorical_crossentropy` for multi-class classification), optimizer (`adam` for efficient gradient descent), and metrics (`accuracy` to monitor training progress).

2. **Building:**
   - Finalizes the model architecture using `model.build()`.
   - Allocates memory for the model's trainable parameters.

3. **Training:**
   - Trains the model on the prepared input (`X`) and label (`y`) data using `model.fit()`.
   - Iterates through epochs (training cycles) to optimize the model's weights based on the loss function.
   - The number of epochs can be adjusted based on dataset size and computational resources.

---

**Prediction**

- A function `predict_next_words()` is defined to generate predictions.
- Given a seed text, the function iteratively predicts the next word using the trained model.
- The predicted word is appended to the seed text, and the process repeats for the specified number of words.

---

**Usage**

1. **User Input:**
   - Prompts the user to enter a seed text and the desired number of words to predict.

2. **Prediction Generation:**
   - Calls the `predict_next_words()` function to generate the predicted sequence.

3. **Output:**
   - Displays the generated text, including the seed text and the predicted words.

---

**Conclusion**

This project demonstrates the successful application of deep learning techniques, particularly LSTM networks, for next word prediction. The model effectively learns from the training data and generates plausible next words, showcasing the potential of deep learning for natural language processing tasks. Further research can explore advancements such as transformer models, larger datasets, and attention mechanisms to enhance prediction accuracy and explore broader applications.

---
