# Sarcasm-Prediction

<h>The goal of this work is to make a classifier model to detect the headlines whether it's a sarcasm or
not.</h>

<b>Steps -</b>
1. Load the train data and test data files using pandas and numpy library.
2. We will need to do some visualization, by this we will know the percentage of both
sarcasm and non-sarcasm headlines in the train data file. For that I have plotted
piechart.
3. Now to do preprocessing of data for only one time, I have merged both train and test
data file (only article_link column and headline column)
4. To remove punctuation, digits and numbers i.e. text cleaning is done.
5. Now tokenization process is done on the cleaned data.
(Tokenization - It is the process of breaking a stream of text up into words, phrases,
symbols, or other meaningful elements called tokens. It's an important process to do in
natural language processing since tokenized words will help for words checking or
conversion process.)
 6. Now lemmatization process is done.
(Lemmatization is a process to converting the words of a sentence to its dictionary form,
which is known as the lemma. Unlike stemming, lemmatization depends on correctly
identifying the intended part of speech and meaning of a word in a sentence, as well as
within the larger context surrounding that sentence, such as neighboring sentences or even
an entire document.)
 7. Now preprocessing should be done by keras preprocessing library to prepare the data.
Before we start the training process with Keras, we need to convert our data so Keras can
read and process it. First we should vectorize our data and convert them into sequences.
 8. Next I need to split the data into training data and testing data. Here I use X1 as a list of data
value and Y to list of prediction value.
 9. In this project I will using LSTM model. There are also some variables that called
hyperparameters which I must set before I train my model and their values are somehow
intuitive, no strict rules or standards to set the values. They are embed dim, number of
neurons, and dropout rate.
 10. Model is build up and it is ready for the training process.
 11. After I trained my model now I can test my model by count it's accuracy.
 12. It’s time to test the model and can be used for prediction purpose.
