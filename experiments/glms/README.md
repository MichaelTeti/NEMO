Here are implementations of different generalized linear models (glms) we use for spatio-temporal receptive field mapping and trace prediction. 

# Elastic Net 
In the [elastic_net.py](https://github.com/MichaelTeti/NEMO/blob/main/experiments/glms/elastic_net.py) script, we use [sklearn's elastic net regression with cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html) with 9 consecutive frames as input and the trial-averaged trace for the 9th frame as the response. Elastic net regression minimizes the objective function 
