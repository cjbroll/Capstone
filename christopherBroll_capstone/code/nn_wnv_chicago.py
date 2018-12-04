# ---------------------------------------------------------------------------
# NOTE: program inspired by Dr. Amir Jafari and Dr. Yuxiao Huang at GWU
# ---------------------------------------------------------------------------
import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, auc, roc_curve

# ----------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

# Read clean dataset
df = pd.read_excel("clean_wnv_chicago_data.xlsx")

# Split into features and targets
X = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values.ravel()

# Split train, validation and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)


# Standardize data
std_x = StandardScaler()

X_train_std = std_x.fit_transform(X_train)
X_test_std = std_x.transform(X_test)
X_val_std = std_x.transform(X_val)


# Change numpy to tensor
p = Variable(torch.from_numpy(np.array(X_train_std, dtype=np.float32)))
t = Variable(torch.from_numpy(y_train)).type(torch.LongTensor)

p_val = Variable(torch.from_numpy(np.array(X_val_std, dtype=np.float32)))
t_val = Variable(torch.from_numpy(y_val)).type(torch.LongTensor)

p_test = Variable(torch.from_numpy(np.array(X_test_std, dtype=np.float32)))
t_test = Variable(torch.from_numpy(y_test)).type(torch.LongTensor)

# -------------------------------------------------------------------------------------------------------
# COMPUTATIONAL GRAPH FOR MODEL
# -------------------------------------------------------------------------------------------------------
# MLP model
model = torch.nn.Sequential(
    torch.nn.Linear(8, 7),
    torch.nn.ReLU(),
    torch.nn.Linear(7, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 2),
    torch.nn.Tanh(),
)

# ---------------------------------------------------------------------------------------------------------
# HYPER PARAMETERS AND OPTIMIZATION
# ---------------------------------------------------------------------------------------------------------

# Imbalanced dataset where 5% of target are 1's
class_weights = torch.FloatTensor([1.0, 9.5])

# Initialize performance index
performance_index = torch.nn.CrossEntropyLoss(weight=class_weights)

# Initialize learning rate
learning_rate = 1e-3

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
epochs = 1000
loss_train = []
loss_val = []
for index in range(epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
            a = model(p)
            loss = performance_index(a, t)
            loss_train.append(loss)
            print(index, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.train(False)
            a = model(p_val)
            loss = performance_index(a, t_val)
            loss_val.append(loss)

# ----------------------------------------------------------------------------------------
# VALIDATION OF MODEL
# ----------------------------------------------------------------------------------------
model.eval()
l = []
for i in range(y_test.shape[0]):
    outputs = model(p_test[i])
    predicted = torch.argmax(outputs.data).item()
    l.append(predicted)

roc_score = roc_auc_score(y_test, l)
print(roc_score)

# -----------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------

# Code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, l)
roc_auc = auc(fpr, tpr)

# Plot roc-curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot loss for training and validation
p1 = plt.plot(range(epochs), loss_train)
p2 = plt.plot(range(epochs), loss_val)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend((p1[0], p2[0]), ('Train', 'Validation'))
plt.title('Performance of MLP')
plt.show()