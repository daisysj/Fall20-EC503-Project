import lsa_kaggle1 as lsa
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


selected_data = []
data = pd.read_csv('./voted-kaggle-dataset.csv')
col_1 = data['Title']
col_2 = data['Subtitle']
for i in range(len(col_1)):
    full_row = str(col_1[i]) + " " + str(col_2[i])
    selected_data.append(full_row)
    full_row = ""
text = lsa.process_data(selected_data)

wordcloud = WordCloud(background_color='white',max_words=100).generate(str(text))
fig = plt.figure(figsize=[30,30])
plt.title('Words in Upvoted Dataset (Preprocessed)')
plt.axis('off')
plt.imshow(wordcloud)
plt.show()
