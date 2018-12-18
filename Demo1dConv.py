from numpy import random, round
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras import optimizers

x_train = round(random.rand(100,5,1),0)
y_train = round(random.rand(100,1),0)
x_train = x_train.astype('float32')
y_train = to_categorical(y_train, num_classes=2)

filters = 10
kernel_size = 5

model = Sequential()
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(5, 1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

history = model.fit(x_train, y_train, epochs=50, batch_size=15, verbose=0)

model.summary()

loss, acc = model.evaluate(x_train, y_train)
print('\nTraining loss: %.2f, acc: %.2f%%'%(loss, acc))
