import numpy as np

X_train = np.load('data_train.npy')
y_train = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)

# Ajouter une colonne de 1 à X_train pour inclure le biais dans les calculs
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Ajoute une colonne de 1 au début

# Calculer theta = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Séparer w et b
b = theta[0]       # Premier élément de theta
w = theta[1:]      # Le reste est w

print("Vecteur w:", w)
print("Biais b:", b)

def __init__(self, h):
        self.h = h

def fit(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.X_train = train_inputs
        self.y_train = train_labels

def predict(self, test_data):
    def onehot_vector(label, count_label):
        onehot_vector = np.zeros(count_label)
        onehot_vector[label-1] = 1

        return onehot_vector
    def weight(self, distance):
        return 1 if distance < self.h else 0

    unique_classes = np.unique(self.y_train)
    unique_classes_counts = len(unique_classes)

    prediction_array = []
    for x in test_data:
        distances = np.sum(np.abs(self.X_train - x), axis=1) #retourne un vecteur des distances entre x et tous les points d'entrainement

    weights = np.array([weight(self, dist) for dist in distances]) #retourne un vecteur du poids de tous les points d'entrainement par rapport a x

    onehot_vect = []
    for y in self.y_train:
        onehot = onehot_vector(int(y), unique_classes_counts)
        onehot_vect.append(onehot)

    if np.sum(weights) != 0:
        weighted_sum = np.sum(weights[:, np.newaxis] * onehot_vect, axis=0)
        weighted_sum /= np.sum(weights)
        prediction_index = np.argmax(weighted_sum)
        prediction = int(unique_classes[prediction_index])
    else:
        prediction = int(draw_rand_label(x, self.y_train))

    prediction_array.append(prediction)
        
    return prediction_array