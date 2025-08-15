### Eye for Blind

1. A notebook to create an Image2Text model based on flicker dataset
2. A notebook to host streamlit ui where an image can be uploaded and the caption is read as audio
3. h5 files for Model Weights 
4. A pkl file for Extracted Features
5. A pkl file for tokenized word index
6. A resultant mp3 file
    
### DataSet :  

/kaggle/input/flickr-image-dataset

### Architecture:

1. Vision Transformer Feature Extractor 
2. Vision Transformer Model for patch embedding
3. Feature Cache to pkl
4. Custom Decoder (with LSTM and Attention Transformers)
5. Inference via Greedy Search
6. Inference via Beam Search
7. Visualizing Attention Weights
8. Bleu Score Eval
9. Model Weights Saved
10. Convert Caption to mp3 using gtts

### UI

1. streamlit
2. ngrok

### Output Samples
![ImageCapionReader](https://github.com/user-attachments/assets/1bdeabce-d81e-4ff8-8c92-13829c0b5257)

[Image_Caption_Reader.pdf](https://github.com/user-attachments/files/21795784/Image_Caption_Reader.pdf)




