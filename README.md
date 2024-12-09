# Discrimination of neutron and gamma-rays based on brain-like neural networks

## Introduction

This project explores the application of two innovative brain-inspired neural networks: the Random-Coupled Neural Network (RCNN) and the Continuous-Coupled Neural Network (CCNN), specifically designed for pulse shape discrimination tasks. By leveraging the unique biological interpretability and temporal structures found in these networks, both RCNN and CCNN can effectively capture the dynamic characteristics of pulse shapes through neural activity, enabling them to perform discrimination without the need for extensive training.

This capability offers a significant advantage, as it reduces the computational resource demands often associated with traditional second-generation neural networks. In our research, we created multiple energy threshold sub-datasets from the time-of-flight (TOF) dataset derived from PuBe sources. Experimental results indicate that both RCNN and CCNN exhibit commendable discrimination efficacy and high accuracy across various energy thresholds, outperforming traditional methods.


![](psd_comparison.jpg)


## Dependencies
* Python 3.11

* numpy

* pandas

* scipy

* skimage

* imageio

* matplotlib

* tqdm

* sklearn

## 1. How to create neutron and gamma datasets with different energy thresholds

   You can directly run **` TOF_NE213A.py `** in the datasetmaker folder to generate TOF datasets at different energy thresholds based on **` NE213A_data.pkl `**. It is worth noting that you need to modify the energy lower limit in line 18 of the **` TOF_NE213A.py `** to specify the threshold energy. After the file runs, a folder named NE213A_* Mev (* represents energy threshold) will be generated, which contains two files, **`gamma_*Mev.npy`** and **`neutron_*Mev.npy`**, representing neutron gamma data at the * Mev energy threshold.

   Since the algorithms used in the experiment are all built on Matlab, it is necessary to convert the npy file into a mat file that can be processed by Matlab through the file **`dataset_maker.m`** located in the folder matlabdataset. Finally, we removed some obvious noise signals through the file **`data_filter.m`** , and saved the processed signal and its normalized version. These signals were also directly processed by the algorithm.
   
   You can directly download npy files and mat files with different energy thresholds and origin pkl file here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14305929.svg)](https://doi.org/10.5281/zenodo.14305929).

## 2. How to use RCNN and CCNN to discriminate datasets

You can directly run **`rcnntest.m`** and **`ccnntest.m`** in the matlabdataset folder to discriminate datasets based on RCNN and CCNN model.

## 3. How to use optuna to tune model parameters
Optuna is an open-source automated hyperparameter optimization framework designed to help developers and researchers efficiently tune the hyperparameters of multiple types of models models [2]. It offers a simple yet powerful way to find the best combinations of hyperparameters, thereby enhancing model performance.

Taking RCNN as an example, you can directly run the **`op_main_rcnn.py`** file to adjust parameters. If there are other parameters that need to be adjusted, they can be specified in the **`objective`** function, taking the af as an example.
   ```bash
   af = trial.suggest_float("af", 0.1, 0.9, step=0.05)
   ```

[2] Akiba, Takuya, et al. "Optuna: A next-generation hyperparameter optimization framework." Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2019.



