'''
Helps visualize the data being fed to the models
'''
import streamlit as st
import numpy as np

from few_shot.constants import OMNIGLOT_SHAPE, CAR_SHAPE, CAR_BATCH_SIZE
from few_shot.data.omniglot_data_loader import omni_data_generator
from few_shot.data.car_data_loader import get_car_generators
from few_shot.models.siamese_model import get_siamese_model


def render_page(generator, model=None):
    '''
    Renders the visualization page
    '''
    st.title('Visualizing Data Input')

    # simply visualize one batch and make sure everything is working properly
    for batch in generator:
        for i in range(32):
            features, labels = batch[0], batch[1]
            is_match = labels[i] == 0
            pair_a = features[0][i]
            pair_b = features[1][i]

            if model is not None:
                prediction = model.predict(
                    [np.array([pair_a]),
                     np.array([pair_b])])[0][0]
            else:
                prediction = 0.0

            is_correct = False
            if (is_match and prediction > 0.5) or (not is_match
                                                   and prediction <= 0.5):
                is_correct = True

            st.write('Index: {} | {} | Prediction: {:.2f} | {}'.format(
                i, 'Match!' if is_match else 'Not match', prediction,
                'correct!' if is_correct else 'WRONG'))

            st.image(pair_a, 'A')
            st.image(pair_b, 'B')

            st.write('-' * 80)

        break


if __name__ == '__main__':
    dataset = 'cars'
    model = None

    if dataset == 'omniglot':
        if False:
            model = get_siamese_model(OMNIGLOT_SHAPE, encoder='lenet')
            model.load_weights(
                './model_checkpoints/omniglot_adam_3.0e-4_switch_label_95.h5')

        generator = omni_data_generator()
        render_page(generator, model)

    if dataset == 'cars':
        train, _, _ = get_car_generators(CAR_BATCH_SIZE, (245, 256, 3))
        render_page(train, model)
