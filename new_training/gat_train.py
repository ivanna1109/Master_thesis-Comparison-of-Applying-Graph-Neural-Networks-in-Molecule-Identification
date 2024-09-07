import sys
import os

# Dodajte putanju do glavnog direktorijuma
sys.path.append('/home/ivanam/Master-thesis')

import new_data_preprocess.read_data as read_data
import metrics.calculate_metrics as metric
from new_training.models.gat import GATModel
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from sklearn.metrics import average_precision_score
from spektral.data import BatchLoader
from spektral.data import Dataset, Graph
from spektral.models import GeneralGNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import math
import numpy as np
from sklearn.metrics import classification_report
import metrics.oversampling_graphs as oversampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
#import evaluate_model as eval



#pribavljanje podataka
graphs, labels = read_data.read_molecule_data_moreActive()
#class_counts = oversampling.count_classes(labels)
#print(f"Count per classes: {class_counts}")
print(f"Podaci su pribavljeni! Velicina grafova: {len(graphs)} velicina labela: {len(labels)}")
classes_num = 3

# trening i test skupovi (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42, stratify=labels)
print("Podaci su podeljeni na trening i test.")

# Dalja podela na trening i validacioni skup (80% trening, 20% validacija)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify = y_train)
print("Podaci su podeljeni na trening i val.")
print(f"Velicina x_train {len(X_train)}")
print(f"Velicina x_val {len(X_val)}")
print(f"Velicina x_test {len(X_test)}")

# Konvertovanje oznaka u one-hot format
y_train = to_categorical(y_train, num_classes=classes_num)
y_val = to_categorical(y_val, num_classes=classes_num)
y_test = to_categorical(y_test, num_classes=classes_num)

print("One hot enkoding labela zavrsen.")

class MyDataset(Dataset):
    def __init__(self, graphs, labels, **kwargs):
        self.graphs = graphs
        self.labels = labels
        print(f"Loaded {len(self.graphs)} graphs")
        super().__init__(**kwargs)

    def read(self):
        output = []
        for i in range(len(self.graphs)):
            output.append(
            Graph(x=self.graphs[i].x, a=self.graphs[i].a, y=self.labels[i],
                  molecule_name = self.graphs[i].get('molecule_name')))
        return output

print("Kreiranje dataset instanci na osnovu podeljenih podataka..")
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

print(f"Broj labela u train dataset: {len(train_dataset.labels)}")
print(f"Broj grafova u train datasetu: {len(train_dataset.graphs)}")

batch_size = 21
print("Prelazimo na batch loadere...")
train_loader = BatchLoader(train_dataset, batch_size=batch_size)
val_loader = BatchLoader(val_dataset, batch_size=batch_size)
test_loader = BatchLoader(test_dataset, batch_size=batch_size)

batch = train_loader.__next__()
inputs, target = batch
x, a = inputs
print(x.shape)
print(a.shape)
print(target.shape)

for batch in train_loader:
    x_batch, l_batch = batch
    print("Dimenzije x_batch0:", x_batch[0].shape)
    print("Dimenzije x_batch1:", x_batch[1].shape)
    print("Dimenzije a_batch:", l_batch.shape)
    break

print("Tri skupa prilagodjena za trening, val i test kreirani da bi mogli pustiti u GCN.")
num_features = 117  # Broj feature-a po čvoru
num_labels = 3     # Broj klasa
n_heads = 8
units = 21
dropout_rate = 0.21
learning_rate = 0.00065
batch_size = 21
    
   
model = GATModel(units, n_heads, dropout_rate, classes_num)

print("Kreiran GAT model.")

# Dodajemo LearningRateScheduler i EarlyStopping sheduler za monitoring treninga
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#custom call back za pracene f1 val score-a
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_f1 = logs.get('val_f1_score')
        if val_f1 is not None:
            print(f'\nEpoch {epoch + 1}: val_f1_score = {val_f1:.4f}')

f1_score_callback = F1ScoreCallback()
print("Dodati LR i ES i custom f1 score")

model.compile(tf.keras.optimizers.Adam(learning_rate=0.00072), #lr po optuna proracunu
              loss='categorical_crossentropy', 
              metrics=['accuracy', 
                       metric.f1_score,
                       AUC(name='roc_auc', multi_label=True),
                    tf.keras.metrics.AUC(curve='PR', name='average_precision')])
                    #metric.weighted_f1_score,
                    #metric.balanced_accuracy

print("Model kompajliran.")

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights_dict = dict(enumerate(class_weights))

train_steps_per_epoch = math.ceil(len(X_train) / batch_size)
val_steps_per_epoch = math.ceil(len(X_val) / batch_size)

print("Start treninga...........................")
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs = 50,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    class_weight = class_weights_dict,
    callbacks=[lr_scheduler, early_stopping, f1_score_callback]
)

print("Zavrsen trening....")
print("Pisemo sumarry modela u fajl......")

with open('new_training/train_results/gat_new/model_summary-gat_moreActiveABCDE.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Zavrseno pisanje u fajl.")

print("Pisemo history treninga..")
history_df = pd.DataFrame(history.history)
history_df.to_csv('new_training/train_results/gat_new/gat_history_moreActiveABCDE.csv', index=False)

print("Cuvanje tezina modela..")
model.save_weights('new_training/train_results/gat_new/gat_model_moreActiveABCDE.h5')

print("Evalucija modela....")
steps_for_test = math.ceil(len(X_test) / batch_size)

test_loss, test_acc, test_f1_score, test_roc_auc, test_avg_precision = model.evaluate(test_loader.load(), batch_size= batch_size, steps =steps_for_test, verbose=1)


print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1Score: {test_f1_score:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")
print(f"Test Average Precision: {test_avg_precision:.4f}")

print("Idemo na classification report....")

y_true_list = []
y_pred_list = []

for step, batch in enumerate(test_loader):
    if step >= steps_for_test:
        break
    inputs, target = batch
    x, a = inputs
    y_true_batch = target # Pretvaranje u numpy array ako već nije
    y_pred_batch = model.predict_on_batch((x, a))
    y_pred_batch = np.argmax(y_pred_batch, axis=-1)
    
    # Ako je y_true_batch jedan-hot enkodiran, koristi np.argmax za pretvaranje
    if len(y_true_batch.shape) > 1 and y_true_batch.shape[1] > 1:
        y_true_batch = np.argmax(y_true_batch, axis=-1)
    
    y_true_list.extend(y_true_batch)
    y_pred_list.extend(y_pred_batch)


print(len(y_true_list))
print(y_true_list[0])
print(len(y_pred_list))
print(y_pred_list[0])
y_true = y_true_list
y_pred = y_pred_list


print(classification_report(y_true, y_pred))



print("Sve uspesno zavrseno.")


