import numpy as np
import skimage
from keras.utils import to_categorical
import skimage.transform
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder=LabelEncoder()):
        self.encoder = encoder

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self
    
    def transform(self, X):
        X = X.copy()
        X = to_categorical(self.encoder.transform(X))
        return X

def _im_resize(df, n, image_size):
    im = skimage.io.imread(df[n])
    im = skimage.transform.resize(im, (image_size, image_size, 3))
    return im

class CreateDataset(BaseEstimator, TransformerMixin):
    def __init__(self, image_size=50):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        tmp = np.zeros(
            (len(X), self.image_size, self.image_size, 3),
            dtype='float32'
        )

        for n in range(0, len(X)):
            im = _im_resize(X, n, self.image_size)
            tmp[n] = im

        print(f"Dataset images shape: {tmp.shape} size: {tmp.size}")
        return tmp
    
if __name__=='__main__':
    import data_management as dm
    import config

    images_df = dm.load_image_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)

    enc = TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    print(y_train)
    print(X_train.head(5))

    dataCreator = CreateDataset()
    x_train = dataCreator.transform(X_train)
