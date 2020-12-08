Here are implementations of different generalized linear models (glms) we use for spatio-temporal receptive field mapping and trace prediction. 

# Elastic Net 
In the [elastic_net.py](https://github.com/MichaelTeti/NEMO/blob/main/experiments/glms/elastic_net.py) script, we use [sklearn's elastic net regression with cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html) with 9 consecutive frames as input and the trial-averaged trace for the 9th frame as the response. Elastic net regression minimizes the objective function (credit to [Mustafa Qamaruddin](https://medium.com/sci-net/cross-validation-strategies-for-time-series-forecasting-9e6cfab91f60) for the image) ![elasticnet_obj](https://github.com/MichaelTeti/NEMO/blob/main/experiments/glms/figures/elasticnet_obj.png) where the first term represents the error between desired and 
predicted responses and the second term is the reguralization term, weighted by lambda (which is called alpha in sklearn's implementation). The regularization
term consists of both a ridge (L2) and lasso (L1) penalty on the coefficients, where alpha (called l1_ratio in sklearn's implementation) determines the tradeoff
between the two. If alpha was 0 you would have ridge regression, and if it was 1, you would have lasso regression. Since elasticnet has the L1 penalty (assuming 
alpha was not 0) a procedure called coordinate descent is used to minimize the objective, as opposed to gradient descent which could be used in ridge regression. 
One of the main differences between coordinate descent and gradient descent is that the parameters are updated one at a time while all others are held constant (to see an implementation of elastic net from scratch, check out [this notebook](https://github.com/MichaelTeti/CAP5625/blob/main/CAP5625_Assignment2_ElasticNet_CoordinateDescent.ipynb)).  
The elastic_net.py script has the following arguments:
```
usage: elastic_net.py [-h] [--write_rf_images] [--write_mse_plots]
                      [--n_frames_in_time N_FRAMES_IN_TIME] [--n_jobs N_JOBS]
                      [--max_iter MAX_ITER] [--n_alphas N_ALPHAS]
                      [--min_l1_ratio MIN_L1_RATIO]
                      [--max_l1_ratio MAX_L1_RATIO]
                      [--n_l1_ratios N_L1_RATIOS]
                      trace_dir stimulus_dir save_dir

positional arguments:
  trace_dir             Path containing the .txt files with trial-averaged
                        traces for a single session type and stimulus.
  stimulus_dir          Path to the stimulus templates corresponding to the
                        trace_dir.
  save_dir              Directory to save results in.

optional arguments:
  -h, --help            show this help message and exit
  --write_rf_images     If specified, write the receptive fields out as
                        images/.gifs.
  --write_mse_plots     If specified, write the mse grid path as a plot.

model parameters:
  Parameter settings for the model and cross-validation.

  --n_frames_in_time N_FRAMES_IN_TIME
                        The number of consecutive video frames to comprise a
                        single input.
  --n_jobs N_JOBS       Number of jobs for the model.
  --max_iter MAX_ITER   The maximum number of iterations for the model.
  --n_alphas N_ALPHAS   The number of alpha (aka lambda) values to search
                        over.
  --min_l1_ratio MIN_L1_RATIO
                        The minimum l1_ratio to try in the grid search.
  --max_l1_ratio MAX_L1_RATIO
                        The maximum l1_ratio to try in the grid search.
  --n_l1_ratios N_L1_RATIOS
                        The number of l1_ratios to try in the range
                        [min_l1_ratio, max_l1_ratio].
``` 
The command we use to run the script is as follows:
```
python3 elastic_net.py \
    ../../data/BrainObservatoryData/ExtractedData/TrialAveragedTraces/natural_movie_three/three_session_A/ \
    ../../data/BrainObservatoryData/ExtractedData/Stimuli/natural_movie_three_resized/ elasticnet_rfs_natural-movie-three \
    --write_rf_images \
    --n_jobs 24 \
    --max_iter 5000 \
    --n_alphas 35 \
    --write_mse_plots \
    --min_l1_ratio 1e-6 \
    --max_l1_ratio 1.0 \
    --n_l1_ratios 6
```
