# Classification Approaches for Covid-19 Fake News Detection
## Comparing various classification approaches to Covid-19 fake news detection

Results:

| Metric    | Naive Bayes | Gradient Boost | Bidirectional LSTM |
|-----------|-------------|----------------|--------------------|
| Accuracy  | 0.890       | 0.852          | **0.897**          |
| Precision | 0.897       | 0.879          | **0.916**              |
| Recall    | **0.883**       | 0.821          | 0.878              |
| F1-Score  | 0.890       | 0.850          | **0.896**              |


If you would like to use the models please download the pickled objects and model checkpoint here: https://drive.google.com/drive/folders/1R2OfcQbjRfHUkrozfAQMv7Q9-WT8V8Nk?usp=sharing 


# How to run

Please create a virtual environment and install the dependencies by running 
```bash
pip install -r requirements.txt
```

Launch Jupyter and view the `Covid-Fake-News.ipynb` file. 

To load the models into memory please save the models to the same location as your Python file or Jupyter notebook and run:
```python
import joblib
# Load the TD-IFD Vectorizer

# Load Naive bayes model
nb_model = joblib.load('naive_bayes.pkl')
# Load Gradient Boosting model
gb_model = joblib.load('gradient_boosting.pkl')

# Load Bidirectional LSTM 
def create_model():
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=32,
            mask_zero=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=["accuracy"])
    return model# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('bilstm/my_checkpoint')
    
```
