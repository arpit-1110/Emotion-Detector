from CNN import CNNmodel
import pandas as pd
import numpy as np
import keras.optimizers as opt

data = pd.read_csv('fer2013/fer2013.csv')

# learning_rate = float(input("Enter Learning Rate: "))
epochs = int(input("Enter number of epochs: "))
# batch_size = int(input("Enter batch size: "))

for i in range(5):
    print()

data = np.array(data)

output = data[:, 0]
y = np.zeros((len(output), 7))
for i in range(len(output)):
    y[i][output[i]] = 1

X = data[:, 1]
X = [np.fromstring(x, dtype='int', sep=' ') for x in X]

X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(48, 48, 1)
              for x in data[:, 1]])

print(X.shape)

X_train, X_test, X_validation = X[:int(0.80 * len(X))], X[:int(
    0.90 * len(X)) - int(0.80 * len(X))], X[:int(len(X)) - int(0.90 * len(X))]
y_train, y_test, y_validation = y[:int(0.80 * len(X))], y[:int(
    0.90 * len(X)) - int(0.80 * len(X))], y[:int(len(X)) - int(0.90 * len(X))]


model = CNNmodel(num_emotions=7)
optimizer = opt.Adadelta()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'],
              )

model.fit(X_train, y_train, validation_data=(
    X_validation, y_validation), epochs=epochs, verbose=2, batch_size=200)

scores = model.evaluate(X_test, y_test, verbose=0)
print(scores[1] * 100)
model_json = model.to_json()
with open("model.json", "w") as f:
    f.write(model_json)
model.save_weights("model.h5")
