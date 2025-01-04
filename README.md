# IMDB Movie Review Sentiment Analysis Using RNN

This project uses a Recurrent Neural Network (RNN) to classify IMDB movie reviews as positive or negative. The model is trained on 50,000 reviews (25,000 for training and 25,000 for testing), achieving an accuracy of 98%.

## Project Structure

The project contains the following files:

1. **requirements.txt**  
   Contains all the dependencies required to run the project.

2. **prediction.ipynb**  
   This Jupyter notebook is used for predicting the sentiment (positive or negative) of movie reviews using the trained RNN model.

3. **EmbeddingTrail.ipynb**  
   This notebook demonstrates how embeddings work, used to represent words in a dense vector format for feeding into the model.

4. **RNN-imdb-review.ipynb**  
   The main Jupyter notebook where the RNN model is defined, trained, and evaluated on the IMDB dataset. The model is trained for 10 epochs with a batch size of 32.

5. **main.py**  
   A Python script used to deploy the trained model using Streamlit for a simple web app interface.

6. **RNN-imdb-review-model.h5**  
   The trained model saved in H5 format, which is used in the prediction and Streamlit app for inference.

## Model Details

- **Model Type**: Recurrent Neural Network (RNN)
- **Dataset**: IMDB movie reviews dataset (25,000 training and 25,000 test samples)
- **Epochs**: 10
- **Batch Size**: 32
- **Accuracy**: 98% on test data
- **Saved Model**: `RNN-imdb-review-model.h5`

## Requirements

To run the project, install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Usage
1. Running the Jupyter Notebooks
You can open the following notebooks to see the workflow:

- EmbeddingTrail.ipynb: Explore how word embeddings are generated and utilized.
- RNN-imdb-review.ipynb: Train and evaluate the RNN model on the IMDB dataset.

2. Prediction with Trained Model
To make predictions on new reviews, use prediction.ipynb or run the following command:
```streamlit run main.py```

This will start a Streamlit web application where you can input movie reviews and get the sentiment (positive/negative) in real-time.

## Try out the Model
https://rnnimdb-movie-review-model-eyek9jtbyxnhyqwi8hnmtb.streamlit.app/

<img width="770" alt="Screenshot 2025-01-04 at 9 27 29 PM" src="https://github.com/user-attachments/assets/36d2d5a5-4761-4ee8-a0b7-39bab01cabc1" />
