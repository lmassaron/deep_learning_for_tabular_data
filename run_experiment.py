# -*- coding: utf-8 -*-
"""
Main script to run the Amazon Employee Access Challenge experiment.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import keras
from keras import ops
from keras.models import Model
from keras.layers import (Dense, BatchNormalization, Dropout, LeakyReLU, Flatten, 
                          Input, Embedding, Concatenate, SpatialDropout1D, Activation)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
import torch
import matplotlib.pyplot as plt

from tabular import TabularTransformer, DataGenerator, set_device, gelu, Mish, mish

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------

print("Loading data...")
X = pd.read_csv('amazon-employee-access-challenge/train.csv')
Xt = pd.read_csv('amazon-employee-access-challenge/test.csv')

y = X["ACTION"].apply(lambda x: 1 if x == 1 else 0).values
X.drop(["ACTION"], axis=1, inplace=True)

print("Label encoding categorical features...")
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
for col, column in enumerate(X.columns):
    label_encoders[col].fit(pd.concat([X[column], Xt[column]]))
    X[column] = label_encoders[col].transform(X[column])
    Xt[column] = label_encoders[col].transform(Xt[column])

print("Frequency encoding features...")
def frequency_encoding(column, df, df_test=None):
    frequencies = df[column].value_counts().reset_index()
    frequencies.columns = ['index', 'counts']
    df_values = df[[column]].merge(frequencies, how='left', 
                                   left_on=column, right_on='index')['counts'].values
    if df_test is not None:
        df_test_values = df_test[[column]].merge(frequencies, how='left', 
                                                 left_on=column, right_on='index')['counts'].fillna(1).values
    else:
        df_test_values = None
    return df_values, df_test_values

for column in X.columns:
    train_values, test_values = frequency_encoding(column, X, Xt)
    X[column+'_counts'] = train_values
    Xt[column+'_counts'] = test_values

categorical_variables = [col for col in X.columns if '_counts' not in col]
numeric_variables = [col for col in X.columns if '_counts' in col]

print("Data shapes:")
print("X train:", X.shape)
print("X test:", Xt.shape)

# -----------------------------------------------------------------------------
# XGBoost Model
# -----------------------------------------------------------------------------

print("\n--- Training XGBoost Model ---")
SEED = 42
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(Xt))
xgb_roc_auc = list()
xgb_average_precision = list()

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_estimators=500, random_state=SEED, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    fold_preds = model.predict_proba(X_test)[:,1]
    xgb_oof[test_idx] = fold_preds
    xgb_roc_auc.append(roc_auc_score(y_test, fold_preds))
    xgb_average_precision.append(average_precision_score(y_test, fold_preds))
    xgb_preds += model.predict_proba(Xt[X.columns])[:,1] / FOLDS

print(f"Average cv roc auc score {np.mean(xgb_roc_auc):0.3f} ± {np.std(xgb_roc_auc):0.3f}")
print(f"Average cv roc average precision {np.mean(xgb_average_precision):0.3f} ± {np.std(xgb_average_precision):0.3f}")
print(f"Roc auc score OOF {roc_auc_score(y, xgb_oof):0.3f}")
print(f"Average precision OOF {average_precision_score(y, xgb_oof):0.3f}")

xgb_submission = pd.DataFrame({'id': Xt.id, 'Action': xgb_preds})
xgb_submission.to_csv("xgboost_submission.csv", index=False)
print("XGBoost submission file created.")

# -----------------------------------------------------------------------------
# Deep Learning Model
# -----------------------------------------------------------------------------

print("\n--- Training Deep Learning Model ---")

# Setup device and custom objects
device = set_device()
keras.utils.get_custom_objects().update({'gelu': Activation(gelu)})
keras.utils.get_custom_objects().update({'mish': Mish(mish)})
keras.utils.get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(negative_slope=0.2))})

def tabular_dnn(numeric_variables, categorical_variables, categorical_counts,
                feature_selection_dropout=0.2, categorical_dropout=0.1,
                first_dense = 256, second_dense = 256, dense_dropout = 0.2, 
                activation_type=gelu):
    
    numerical_inputs = Input(shape=(len(numeric_variables),))
    numerical_normalization = BatchNormalization()(numerical_inputs)
    numerical_feature_selection = Dropout(feature_selection_dropout)(numerical_normalization)

    categorical_inputs = []
    categorical_embeddings = []
    for category in  categorical_variables:
        categorical_inputs.append(Input(shape=[1], name=category))
        category_counts = categorical_counts[category]
        categorical_embeddings.append(
            Embedding(category_counts+1, 
                      int(np.log1p(category_counts)+1), 
                      name = category + "_embed")(categorical_inputs[-1]))

    categorical_logits = Concatenate(name = "categorical_conc")([Flatten()(SpatialDropout1D(categorical_dropout)(cat_emb)) 
                                                                 for cat_emb in categorical_embeddings])

    x = Concatenate()([numerical_feature_selection, categorical_logits])
    x = Dense(first_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)  
    x = Dense(second_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model([numerical_inputs] + categorical_inputs, output)
    
    return model

# Custom Metric for Average Precision
class AveragePrecision(keras.metrics.Metric):
    def __init__(self, name="average_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.y_true_flat = []
        self.y_pred_flat = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true_flat.append(ops.reshape(y_true, [-1]))
        self.y_pred_flat.append(ops.reshape(y_pred, [-1]))

    def result(self):
        y_true = ops.concatenate(self.y_true_flat, axis=0)
        y_pred = ops.concatenate(self.y_pred_flat, axis=0)
        return average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    def reset_state(self):
        self.y_true_flat = []
        self.y_pred_flat = []

def compile_model(model, loss, metrics, optimizer):
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model

def plot_keras_history(history, measures, fold_number):
    """
    history: Keras training history
    measures: list of names of measures
    fold_number: the fold number to include in the filename
    """
    rows = len(measures) // 2 + len(measures) % 2
    fig, panels = plt.subplots(rows, 2, figsize=(15, 5))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    try:
        panels = [item for sublist in panels for item in sublist]
    except:
        pass
    for k, measure in enumerate(measures):
        panel = panels[k]
        panel.set_title(measure + ' history')
        panel.plot(history.epoch, history.history[measure], label="Train "+measure)
        panel.plot(history.epoch, history.history["val_"+measure], label="Validation "+measure)
        panel.set(xlabel='epochs', ylabel=measure)
        panel.legend()
        
    plt.savefig(f"fold_{fold_number}_history.png")
    plt.close(fig)

# Training settings
BATCH_SIZE = 512
measure_to_monitor = 'val_auc'
modality = 'max'

# CV Iteration
roc_auc = list()
average_precision = list()
oof = np.zeros(len(X))
best_iteration = list()

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    
    tb = TabularTransformer(numeric=numeric_variables, ordinal=[], lowcat=[], highcat=categorical_variables)
    tb.fit(X.iloc[train_idx])
    sizes = tb.shape(X.iloc[train_idx])
    categorical_levels = dict(zip(categorical_variables, sizes[1:]))
    print(f"Input array sizes: {sizes}")
    print(f"Categorical levels: {categorical_levels}\n")
    
    model = tabular_dnn(numeric_variables, categorical_variables, categorical_levels, 
                        feature_selection_dropout=0.1, categorical_dropout=0.1,
                        first_dense=256, second_dense=256, dense_dropout=0.1,
                        activation_type=gelu)
    
    model = compile_model(model, 'binary_crossentropy', [AUC(name='auc'), AveragePrecision(name='average_precision')], Adam(learning_rate=0.0001))
    
    train_batch = DataGenerator(X.iloc[train_idx], y[train_idx],
                                tabular_transformer=tb, batch_size=BATCH_SIZE,
                                shuffle=True, device=device)
    
    validation_data = (tb.transform(X.iloc[test_idx]), y[test_idx])

    early_stopping = EarlyStopping(monitor=measure_to_monitor, mode=modality, patience=3, verbose=0)
    checkpoint_file = f'best_fold_{fold}.keras'
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor=measure_to_monitor, mode=modality, save_best_only=True, verbose=0)
    
    history = model.fit(train_batch,
                        validation_data=validation_data,
                        epochs=30,
                        callbacks=[model_checkpoint, early_stopping],
                        class_weight={0:1.0, 1:(np.sum(y==0) / np.sum(y==1))},
                        verbose=1)
    
    print(f"\nFOLD {fold+1}")
    plot_keras_history(history, measures=['auc', 'loss'], fold_number=fold)
    
    best_iteration.append(np.argmax(history.history['val_auc']) + 1)
    
    # Load the best model from the checkpoint file
    best_model = keras.models.load_model(
        checkpoint_file,
        custom_objects={
            'gelu': Activation(gelu), 
            'mish': Mish(mish), 
            'AveragePrecision': AveragePrecision
        }
    )
    
    preds = best_model.predict(tb.transform(X.iloc[test_idx]),
                               verbose=1,
                               batch_size=1024).flatten()

    oof[test_idx] = preds
    roc_auc.append(roc_auc_score(y_true=y[test_idx], y_score=preds))
    average_precision.append(average_precision_score(y_true=y[test_idx], y_score=preds))

print(f"Average cv roc auc score {np.mean(roc_auc):0.3f} ± {np.std(roc_auc):0.3f}")
print(f"Average cv roc average precision {np.mean(average_precision):0.3f} ± {np.std(average_precision):0.3f}")
print(f"Roc auc score OOF {roc_auc_score(y_true=y, y_score=oof):0.3f}")
print(f"Average precision OOF {average_precision_score(y_true=y, y_score=oof):0.3f}")

# Final DNN model training
print("\n--- Training Final DNN Model ---")
tb = TabularTransformer(numeric=numeric_variables, ordinal=[], lowcat=[], highcat=categorical_variables)
tb.fit(X)
sizes = tb.shape(X)
categorical_levels = dict(zip(categorical_variables, sizes[1:]))
print(f"Input array sizes: {sizes}")
print(f"Categorical levels: {categorical_levels}\n")

model = tabular_dnn(numeric_variables, categorical_variables, categorical_levels, 
                    feature_selection_dropout=0.1, categorical_dropout=0.1,
                    first_dense=256, second_dense=256, dense_dropout=0.1,
                    activation_type=gelu)
    
model = compile_model(model, 'binary_crossentropy', [AUC(name='auc'), AveragePrecision(name='average_precision')], Adam(learning_rate=0.0001))    

train_batch = DataGenerator(X, y,
                            tabular_transformer=tb,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            device=device)

history = model.fit(train_batch,
                    epochs=int(np.median(best_iteration)),
                    class_weight={0:1.0, 1:(np.sum(y==0) / np.sum(y==1))},
                    verbose=1)

preds = model.predict(tb.transform(Xt[X.columns]),
                      verbose=1,
                      batch_size=1024).flatten()

tabular_dnn_submission = pd.DataFrame({'id': Xt.id, 'Action': preds})
tabular_dnn_submission.to_csv("tabular_dnn_submission.csv", index=False)
print("DNN submission file created.")

# -----------------------------------------------------------------------------
# Blending
# -----------------------------------------------------------------------------

print("\n--- Blending Models ---")
from scipy.stats import rankdata

dnn_rank = rankdata(tabular_dnn_submission.Action, method='dense') / len(Xt)
xgb_rank = rankdata(xgb_submission.Action, method='dense') / len(Xt)

submission = pd.DataFrame({'id': Xt.id, 'Action': 0.5 * dnn_rank + 0.5 * xgb_rank})
submission.to_csv("blended_submission.csv", index=False)
print("Blended submission file created.")
print("Script finished.")
