from kerasify import export_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


test_x = np.random.rand(10, 10, 10).astype('f')
test_y = np.random.rand(10).astype('f')

model = Sequential()
model.add(LSTM(2, input_shape=(10, 10)))
model.add(Dense(1, input_dim=10))

model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit(test_x, test_y, epochs=3, verbose=2)

print model.predict(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))
print('1')

export_model(model, 'example.model')
print('2')