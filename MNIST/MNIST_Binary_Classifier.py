from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict

mnist =  fetch_openml("mnist_784", as_frame = False)
data, target = mnist.data, mnist.target
# for fortunately, the data is already shuffled
train_data, test_data, train_target, test_target = data[:60000], data[60000:], target[:60000], target[60000:]
train_target_5, test_target_5 = (train_target == '5'), (test_target == '5')
# the estimator
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(train_data, train_target_5)
# getting the predictions 
pred_train =  cross_val_predict(sgd_clf, train_data, train_target_5, cv = 3)
# measure our model using confusion matrix
conf_mx = confusion_matrix(train_target_5, pred_train)
precision = precision_score(train_target_5, pred_train) # 83%
recall = recall_score(train_target_5, pred_train) # 65% 

# what if we want to precision to be 90 
scores_train = cross_val_predict(sgd_clf, train_data, train_target_5, cv = 3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(train_target_5, scores_train)

idx_for_90_precision = (precisions >= 0.9).argmax() # argmax to get the first thershold
threshold = thresholds[idx_for_90_precision]

train_pred_90_target = (scores_train >= threshold) 
new_precision = precision_score(train_target_5, train_pred_90_target) # 90%
new_recall = recall_score(train_target_5, train_pred_90_target) # 47% 