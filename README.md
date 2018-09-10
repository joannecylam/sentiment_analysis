# Sentiment Analysis using LSTM
Classify restaurant comments into positive and negative comments.

Dataset: AI Challenger: Restaurant Reviews in Chinese Language
https://challenger.ai/competition/fsauor2018?type=myteam

## Model
keras LSTM with tensorflow backend
_________________________________________________________________
Layer (type)                 Output Shape              Param #   

embedding_1 (Embedding)      (None, 1173, 128)         256000    
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 1173, 128)         0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 196)               254800    
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 394       

## Performance
Accuracy ~ 0.95




