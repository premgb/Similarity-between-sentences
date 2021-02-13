#########################################################################################################################################
# Determine similarity between sentences                                                                                                #
#########################################################################################################################################

#Import library
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text

pd.set_option('display.max_colwidth', None)

#reference sentence
ref_sentence = "Named must your fear be before banish it you can."

#Load data file in which we need to find sentences similar to reference sentence
df_sentence = pd.read_csv("data/yoda.csv")
df_sentence.head()
df_sentence.shape

#Find similarity using Google's Universal sentence encoder
da_sentences = df_sentence["text"].tolist()
en_sentences = [ref_sentence] * df_sentence.shape[0]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

da_result = embed(da_sentences)
en_result = embed(en_sentences)

# Compute similarity matrix
similarity_matrix = np.inner(en_result, da_result)

df_sentence = pd.concat([df_sentence, pd.DataFrame(similarity_matrix[0], columns=['similarity_to_ref_sentence'])], axis=1)

#So sentences similar to reference sentence are...
df_final = df_sentence.sort_values('similarity_to_ref_sentence', ascending=False)
df_final.head(5)
