import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'# INFO messages are not printed

# Loading packages
import os
import sys
import numpy as np
import pandas as pd
import feather

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
import lightgbm as lgb
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Flatten, GlobalAvgPool1D, GlobalMaxPool1D
from tensorflow.keras.layers import Input, Embedding, Reshape, concatenate, Concatenate, MaxPooling1D, Conv1D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support as score
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import json
import joblib

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import SpatialDropout1D

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer

import os
from keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

class Mish(Activation):
    '''
    Mish Activation Function.
    see: https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

class ItemFilterOut():
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        to_drop = [key for key in self.keys if key in data_dict]
        return data_dict.drop(to_drop, axis=1)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
        
class ItemSelector():
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].fillna('UNK')
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class SplitInput():
    
    def __init__(self, sequence):
        self.sequence = sequence
        
    def fit(self, X, y=None):
        for op in self.sequence:
            op.fit(X)
        return self
    
    def transform(self, X, y=None):
        return [op.transform(X) for op in self.sequence]
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)

class LEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.encoders = list()
        self.dictionary_size = list()
        self.unk = -1
    
    def fit(self, X, y=None, **fit_params):
        for col in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(X.iloc[:, col].fillna('_nan'))
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            
            if '_nan' not in le_dict:
                max_value = max(le_dict.values())
                le_dict['_nan'] = max_value
            
            max_value = max(le_dict.values())
            le_dict['_unk'] = max_value
            
            self.unk = max_value
            self.dictionary_size.append(len(le_dict))
            self.encoders.append(le_dict)
            
        return self
    
    def transform(self, X, y=None, **fit_params):
        output = list()
        for col in range(X.shape[1]):
            le_dict = self.encoders[col]
            emb = X.iloc[:, col].fillna('_nan').apply(lambda x: le_dict.get(x, self.unk)).values
            output.append(emb)
        return output

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class ToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.astype(str)
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class TabularTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, numeric=list(), ordinal=list(), 
                 lowcat=list(), highcat=list()):
        
        self.numeric = numeric
        self.ordinal = ordinal
        self.lowcat  = lowcat
        self.highcat = highcat
        
        self.mvi = multivariate_imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=300, n_jobs=-2), 
                                          initial_strategy='median')

        self.uni = univariate_imputer = SimpleImputer(strategy='median', 
                                           add_indicator=True)

        self.nmt = numeric_transformer = Pipeline(steps=[
            ('normalizer', QuantileTransformer(n_quantiles=600,
                                               output_distribution='normal',
                                               random_state=42)),
            ('imputer', univariate_imputer),
            ('scaler', StandardScaler())])

        self.ohe = generic_categorical_transformer = Pipeline(steps=[
            ('string_converter', ToString()),
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.lle = label_enc_transformer = Pipeline(steps=[
            ('string_converter', ToString()),
            ('label_encoder', LEncoder())])
        
        self.ppl = ColumnTransformer(
            transformers=[
        ('num', numeric_transformer, self.numeric+self.ordinal),
        ('ohe', generic_categorical_transformer, self.lowcat+self.ordinal),
    ], remainder='drop')

    def fit(self, X, y=None, **fit_params):
        _ = self.ppl.fit(X)
        if len(self.highcat) > 0:
            _ = self.lle.fit(X[self.highcat])
        return self
    
    def shape(self, X, y=None, **fit_params):
        numeric_shape = self.ppl.transform(X.iloc[[0],:]).shape[1]
        categorical_size = self.lle.named_steps['label_encoder'].dictionary_size
        return [numeric_shape] + categorical_size
    
    def transform(self, X, y=None, **fit_params):
        Xn = self.ppl.transform(X)
        if len(self.highcat) > 0:
            return [Xn] + self.lle.transform(X[self.highcat])
        else:
            return Xn
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, X, y,
                 tabular_transformer=None,
                 batch_size=32, 
                 shuffle=False,
                 dict_output=False
                 ):
        
        'Initialization'
        self.X = X
        self.y = y
        self.tbt = tabular_transformer
        self.tabular_transformer = tabular_transformer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dict_output = dict_output
        self.indexes = self._build_index()
        self.on_epoch_end()
        self.item = 0
    
    def _build_index(self):
        """
        Builds an index from data
        """
        return np.arange(len(self.y))
    
    def on_epoch_end(self):
        """
        At the end of every epoch, shuffle if required
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(len(self.indexes) / self.batch_size) + 1
    
    
    def __iter__(self):
        """
        returns an iterable
        """
        for i in range(self.__len__()):
            self.item = i
            yield self.__getitem__(index=i)
            
        self.item = 0
        
    def __next__(self):
        return self.__getitem__(index=self.item)
    
    def __call__(self):
        return self.__iter__()
            
    def __data_generation(self, selection):
        if self.tbt is not None:
            if self.dict_output:
                dct = {'input_'+str(j) : arr for j, arr in enumerate(self.tbt.transform(self.X.iloc[selection, :]))}
                return dct, self.y[selection]
            else:
                return self.tbt.transform(self.X.iloc[selection, :]), self.y[selection]
        else:
            return self.X.iloc[selection, :], self.y[selection]
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        samples, labels = self.__data_generation(indexes)
        return samples, labels


if __name__ == "__main__":
	pass