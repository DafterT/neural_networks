# %%
import pandas as pd
import numpy as np
from neural_network import Kohonen
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('my_data_norm.csv')
df.head()
# %%
selected_columns = ['ac', 'health', 'strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma']
df_selected = df[selected_columns]
numpy_arrays = df_selected.to_numpy()
numpy_arrays
# %%
kh = Kohonen(len(selected_columns), 400)
kh.generate_W()
# %%
kh.calculate(numpy_arrays, 5)
# %%
kh.W.shape
# %%
fig, ax = plt.subplots(
    nrows=2, ncols=4, figsize=(12, 4), 
    subplot_kw=dict(xticks=[], yticks=[]))

# Initialize the SOM randomly to the same state

for i in range(2):
    for j in range(4):
        new_arr = np.expand_dims(kh.W[..., i * 2 + j], axis=-1)
        new_arr = np.concatenate([new_arr]*3, axis=-1)
        ax[i][j].imshow(new_arr)
        ax[i][j].title.set_text(f'{i * 2 + j}')
