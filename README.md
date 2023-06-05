# Autoreplicative Random Forests

### Imputation.py: 

class **Imputation** for running iterative and procedural imputation methods realised for:
- Autoreplicative Random Forests (iterative, distributional iterative, and procedural)
- Autoencoders (iterative and procedural)
- PCA (iterative and procedural)
- MICE (iterative by definition)

### ProbabilisticRF: 

class **ProbabilisticTreeForest** for the ditARF method


### Example of code running:


```python

### upload dataset containing missing values
X_na = ... 

### placeholder for missing values, default: -1
na = ...  

if mdl_name == 'itARF':
  mdl = Imputation(estimator='RF', method='iterative', na=na)
elif mdl_name == 'pARF':
  mdl = Imputation(estimator='RF', method='procedural', na=na)
elif mdl_name == 'ditARF':
  mdl = Imputation(estimator='ditARF', method='iterative', na=na)
  
### other possibilities: it- and pAE, it- and pPCA, MICE
### hyperparameters of the methods may be also provided
            
X_imps = mdl.impute(X_na=X_na) ### returns imputations from all iterations
X_imp = X_imps[-1] ### take the last imputation

```
