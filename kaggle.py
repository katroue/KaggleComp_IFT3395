import numpy as np

# Chargement des fichiers
X_train = np.load('data_train.npy')  # fichier NumPy avec les caractéristiques d'entraînement
X_test = np.load('data_test.npy')    # fichier NumPy avec les caractéristiques de test
y_train = np.genfromtxt('label_train.csv', delimiter=',')  # fichier CSV avec les étiquettes
y_train = y_train[1:]
# Extract the first column as y_train_id
y_train_id = y_train[:, 0]

# Extract the second column as y_train_labels
y_train_labels = y_train[:, 1]

# Compute term frequencies across all documents
term_frequencies = np.sum(X_train, axis=0)

# Define a threshold for rare or overly common words
rare_threshold = 0.05 * X_train.shape[0]         # Words appearing in less that 0.05% of the time in all the documents
common_threshold = 0.8 * X_train.shape[0]  # Words appearing in more than 80% of documents

# Create a mask to keep words that are neither too rare nor too common
valid_words_mask = (term_frequencies > rare_threshold) & (term_frequencies < common_threshold)

# Filter the X_train and vocab_map based on this mask
X_train_filtered = X_train[:, valid_words_mask]
X_test_filtered = X_test[:, valid_words_mask]

class NaiveBayesBinaryClassifier:
    def __init__(self):
        self.class_prior = {}  # P(c)
        self.word_probs = {}   # P(x|c)

    def fit(self, X_train, y_train, alpha=1.0):
        """
        X_train: numpy array, shape (n_samples, n_features), where n_features is the number of words (bag of words)
        y_train: numpy array, shape (n_samples,), binary labels (0 or 1)
        alpha: smoothing parameter for Laplace smoothing
        """
        n_samples, n_features = X_train.shape
        classes = np.unique(y_train)

        # Calculate prior probabilities P(c)
        self.class_prior = {cls: np.mean(y_train == cls) for cls in classes}

        # Calculate conditional probabilities P(x|c) with Laplace smoothing
        self.word_probs = {}
        for cls in classes:
            X_cls = X_train[y_train == cls]
            self.word_probs[cls] = (np.sum(X_cls, axis=0) + alpha) / (X_cls.shape[0] + alpha * 2)

    def predict(self, X_test):
        """
        X_test: numpy array, shape (n_samples, n_features), where n_features is the number of words (bag of words)
        Returns: predictions, array of shape (n_samples,)
        """
        n_samples = X_test.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            log_probs = {}
            for cls in self.class_prior:
                # Calculating log P(c) + log P(x|c)
                log_prob_cls = np.log(self.class_prior[cls]) + np.sum(X_test[i] * np.log(self.word_probs[cls]))
                log_probs[cls] = log_prob_cls

            # Select the class with the highest log-probability
            predictions[i] = max(log_probs, key=log_probs.get)

        return predictions


model = NaiveBayesBinaryClassifier()
model.fit(X_train_filtered, y_train_labels)

# Prédictions sur l'ensemble de test
y_pred_bayes_naif = model.predict(X_test_filtered) # terminé en 6sec !

# Générer des indices pour l'ensemble de test
test_ids = np.arange(0, len(y_pred_bayes_naif))  # Par exemple, créer des IDs allant de 1 à n_samples

# Créer le tableau avec les IDs et les prédictions
results = np.column_stack((test_ids, y_pred_bayes_naif))

# Sauvegarder les résultats dans un fichier CSV avec les en-têtes
np.savetxt('naive_bayes_predictions_test.csv', results, delimiter=',', fmt='%d', header='ID,label', comments='')
