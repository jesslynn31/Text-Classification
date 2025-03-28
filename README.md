This project was meant as a learning project. I find deep learning and machine learning as a whole super interesting. I wanted to take it 
to the next level and use my GPU to train a model. I used a Rotten Tomatoes movie review dataset, which had about 10,000 pieces of data. 
The data was split 5,000 positive and 5,000 negative just about. 

The dataset from Hugging Face: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes

The model then was trained to group the dataset based off 
language processing skills from the 'distilbert-base-uncased model'. 

It went through 3 phases of training. The performance got better each time. 

![image](https://github.com/user-attachments/assets/497a9885-7441-4ae1-adf4-1689983eab7e)


After the processing, training and evaluating phase, the final results were put into a .csv file. 


To visualization things, the results were put into a confusion matrix and pie chart. 


![image](https://github.com/user-attachments/assets/350173e6-f156-4f62-92db-53f4cbac815f)


![image](https://github.com/user-attachments/assets/d642e8e4-cb6e-4b9f-ac29-bad76f3944dd)



To analyze my data, the model performed 
like this the first time around:


Accuracy: 83.6%

Precision: 86.7%

Recall: 79.7%

F1 Score: 83.0%


