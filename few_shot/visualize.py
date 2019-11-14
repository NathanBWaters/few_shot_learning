'''
Helps visualize the data being fed to the models
'''
import streamlit as st

from few_shot.data.omniglot_data_loader import omni_data_generator
from keras.models import load_model

st.title('Visualizing Data Input')

generator = omni_data_generator('data/images_background', 32, (105, 105, 3))
model = load_model('./model_checkpoints/omniglot_basic_cnn_345.h5')

# simply visualize one batch and make sure everything is working properly
for batch in generator:
    for i in range(32):
        features, labels = batch[0], batch[1]
        st.write('Index: {} | {} | {}'.format(
            i,
            'Match!' if labels[i] == 1 else 'Not match',
            '-' * 80))
        st.image(features[0][i], 'A')
        st.image(features[1][i], 'B')

    break
