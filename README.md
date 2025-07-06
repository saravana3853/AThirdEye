A notebook to caption image and read out the caption , act as an eye for blind people

DataSet :  /kaggle/input/flickr-image-dataset

Architecture:

1. Vision Transformer Feature Extractor 
2. Vision Transformer Model
3. Feature Cache to pkl
4. Custom Decoder (with LSTM and Attention Transformers)
5. Inference via Greedy Search
6. Inference via Beam Search
7. Visualizing Attention Weights
8. Bleu Score Eval
9. Model Weights Saved
10. Convert Caption to mp3 using gtts


