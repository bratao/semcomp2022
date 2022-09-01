from pathlib import Path
import numpy as np
import os

import joblib
from skimage.transform import resize
from collections import Counter
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler
import skimage
from sklearn.pipeline import Pipeline


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


def carregar_imagens(src, include, width=150, height=None):
    """
    carregar imagens do path, redimensionar e retornar um array de imagens

    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'Imagens de bichinhos ({0}x{1}) redimensionadas em rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = src / subdir

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))  # [:,:,::-1]
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)
    return data


if __name__ == '__main__':
    dataset_folder = Path(__file__).parent.parent / 'dataset'
    width = 80
    include = {'cachorro', 'gato'}
    data = carregar_imagens(src=dataset_folder, width=width, include=include)

    print('Número de amostras : ', len(data['data']))
    print('chaves: ', list(data.keys()))
    print('Descrição: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
    print(Counter(data['label']))

    # Vamos carregar as imagens em um array numpy
    X = np.array(data['data'])
    y = np.array(data['label'])

    # Vamos dividir o conjunto em teste e treino, 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    print("Tamanho do conjunto de treino: ", X_train.shape[0])
    print("Tamanho do conjunto de teste: ", X_test.shape[0])

    # Criar nosso classificador
    # HOG-SVM, que significa Histograma de Gradientes Orientados .
    # Os HOGs são usados para redução de feaures, em outras palavras: para diminuir a complexidade do problema,
    # mantendo o máximo de variação possível.
    #

    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()

    # Vamos criar um pipeline para o classificador, assim fica mais facil de treinar e salvar em disco
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)

    hog_pipeline = Pipeline([
        ('grayify', grayify),
        ('hogify', hogify
         ),
        ('scalify', scalify),
        ('classify', sgd_clf)
    ])

    hog_pipeline.fit(X_train, y_train)

    y_pred = hog_pipeline.predict(X_test)
    print(np.array(y_pred == y_test)[:25])
    print('\nPercentual correto: ', 100 * np.sum(y_pred == y_test) / len(y_test))

    print(classification_report(y_test, y_pred))

    path_model = Path(__file__).parent.parent / 'modelo_treinado'
    path_model.mkdir(exist_ok=True)

    # Vamos salvar o pipeline em disco
    joblib.dump(hog_pipeline, path_model / 'hog_pipeline.pkl')



