Here are implementations of different generalized linear models (glms) we use for spatio-temporal receptive field mapping and trace prediction. 

# Elastic Net 
In the [elastic_net.py](https://github.com/MichaelTeti/NEMO/blob/main/experiments/glms/elastic_net.py) script, we use elastic net regression to take in
9 consecutive frames at a time and model the trace observed at the 9th frame. 
