
# Machine Learning Projects

Some Machine Learning Project(s)

## Result of Classification for Social Ads

![Result of Classification](media/Figure_1.png)

## Accuracy of Tumor Classification 
```
-> deep learning 0.9708029197080292
-> logistic regression 0.9562043795620438
-> k-nearest-neighbors 0.9562043795620438
-> svc 0.9562043795620438
-> decision tree 0.9562043795620438
-> naive bayes 0.948905109489051
-> random forest 0.948905109489051
```

## Accuracy of Customer Reviews NLP Classification 
```
('->', 'logistic regression', 0.865)
('->', 'svc                ', 0.85)
('->', 'deep learning      ', 0.825)
('->', 'random forest      ', 0.785)
('->', 'decision tree      ', 0.78)
('->', 'naive bayes        ', 0.745)
('->', 'k-nearest-neighbors', 0.575)
```

### Single Review Prediction
```
########################
 -> custom sentence -> 'the food was amazing'
 -> we expect: True
 ---> prediction 'logistic regression' say True -----> SUCCESS
 ---> prediction 'svc                ' say True -----> SUCCESS
 ---> prediction 'deep learning      ' say True -----> SUCCESS
 ---> prediction 'random forest      ' say True -----> SUCCESS
 ---> prediction 'decision tree      ' say True -----> SUCCESS
 ---> prediction 'naive bayes        ' say True -----> SUCCESS
 ---> prediction 'k-nearest-neighbors' say True -----> SUCCESS
########################
 -> custom sentence -> 'the food was great'
 -> we expect: True
 ---> prediction 'logistic regression' say True -----> SUCCESS
 ---> prediction 'svc                ' say True -----> SUCCESS
 ---> prediction 'deep learning      ' say True -----> SUCCESS
 ---> prediction 'random forest      ' say True -----> SUCCESS
 ---> prediction 'decision tree      ' say True -----> SUCCESS
 ---> prediction 'naive bayes        ' say True -----> SUCCESS
 ---> prediction 'k-nearest-neighbors' say True -----> SUCCESS
########################
 -> custom sentence -> 'the food was good'
 -> we expect: True
 ---> prediction 'logistic regression' say True -----> SUCCESS
 ---> prediction 'svc                ' say True -----> SUCCESS
 ---> prediction 'deep learning      ' say True -----> SUCCESS
 ---> prediction 'random forest      ' say True -----> SUCCESS
 ---> prediction 'decision tree      ' say True -----> SUCCESS
 ---> prediction 'naive bayes        ' say True -----> SUCCESS
 ---> prediction 'k-nearest-neighbors' say True -----> SUCCESS
########################
 -> custom sentence -> 'slow service but great taste'
 -> we expect: True
 ---> prediction 'logistic regression' say True -----> SUCCESS
 ---> prediction 'svc                ' say False -----> FAILURE
 ---> prediction 'deep learning      ' say False -----> FAILURE
 ---> prediction 'random forest      ' say False -----> FAILURE
 ---> prediction 'decision tree      ' say True -----> SUCCESS
 ---> prediction 'naive bayes        ' say False -----> FAILURE
 ---> prediction 'k-nearest-neighbors' say False -----> FAILURE
########################
 -> custom sentence -> 'the food was bad'
 -> we expect: False
 ---> prediction 'logistic regression' say False -----> SUCCESS
 ---> prediction 'svc                ' say False -----> SUCCESS
 ---> prediction 'deep learning      ' say False -----> SUCCESS
 ---> prediction 'random forest      ' say False -----> SUCCESS
 ---> prediction 'decision tree      ' say False -----> SUCCESS
 ---> prediction 'naive bayes        ' say False -----> SUCCESS
 ---> prediction 'k-nearest-neighbors' say False -----> SUCCESS
########################
 -> custom sentence -> 'I'll will not order from them again'
 -> we expect: False
 ---> prediction 'logistic regression' say False -----> SUCCESS
 ---> prediction 'svc                ' say False -----> SUCCESS
 ---> prediction 'deep learning      ' say False -----> SUCCESS
 ---> prediction 'random forest      ' say False -----> SUCCESS
 ---> prediction 'decision tree      ' say False -----> SUCCESS
 ---> prediction 'naive bayes        ' say True -----> FAILURE
 ---> prediction 'k-nearest-neighbors' say False -----> SUCCESS
########################
 -> custom sentence -> 'avoid that place'
 -> we expect: False
 ---> prediction 'logistic regression' say False -----> SUCCESS
 ---> prediction 'svc                ' say False -----> SUCCESS
 ---> prediction 'deep learning      ' say False -----> SUCCESS
 ---> prediction 'random forest      ' say False -----> SUCCESS
 ---> prediction 'decision tree      ' say True -----> FAILURE
 ---> prediction 'naive bayes        ' say False -----> SUCCESS
 ---> prediction 'k-nearest-neighbors' say False -----> SUCCESS
 ```


