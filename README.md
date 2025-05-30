## Introduction
GraphLOGIC: Graph-based Lethality predictor for OsteoGenesis Imperfecta for Collagen is implementation osteogenesis imperfecta of model for structural-informed lethality prediction. The code is published with "Developing clinical lethality predictive model of Osteogenesis Imperfecta by using graph neural network".

Model checkpoint and atomic structure data were available at https://doi.org/10.6084/m9.figshare.24633969. To run the following code, you will have to download the model checkpoint and place them under the same directory.

## Environment

1. **Hardware**

The model is training on an AMD R5 3600 without GPU acceleration and using PyTorch and PyTorch Geometric. You will also need pandas for data processing, as well as Matplotlib for the figures.

2. **Packages**

All of the script are written in Python.
Clone our environment with:
```
conda env create -f environment.yml
```
Then, activate the environment:
```
conda activate GraphLOGIC
```

## Running the Code
First, clone this repo and download the model checkpoints (`bert4_final_d07`, `bert4_final_d15`, `node_embedding`) and place them in the root of this repo. An explanation of all directories is as follows:
- `bert4_final_d07`: a1 model
- `bert4_final_d15`: a1 & a2 model
- `node_embedding`: sequence embedding vector
- `homo_eq`: full atomic simulation data
- `crossvalidation`: cross-validation data
- `dataset`: OI dataset
- `figure`: publishion data
- `Grad-CAM`: grad-cam analysis data
- `ref-2015`: the result of xiao.2015
- `reference_structure`: the structure use to build graph
1. **Reproduce**

To reproduce the results shown in the paper on GraphLOGIC:
```
python detail_information.py
```
Edit the parameters at the beginning of `detail_information.py` to select between GraphLOGIC trained on the a1 or a12 dataset (see our paper for details):
```
# a12 dataset, selected by default
result_type = "shuffle"
result_dataset = "test" #(or total)
dataset_name =  "bert4_total_real"
save_dir = "./bert4_final_d15/"
model_arch = "GAT_n_tot"
t = "a2"

# a1 dataset
result_type = "control"
result_dataset = "test" #(or total)
dataset_name =  "bert4_ref_real"
save_dir = "./bert4_final_d07/"
model_arch = "GAT_n_tot_only"
t = "a1"
```
2. **Training results**

Run the scripts for plotting.
```
python plot_cv.py
python plot_pr_curve.py
python plot_prediction_15-23.py
python plot_tsne.py
```
Figures are saved in the `figures` directory.

3. **Grad-CAM**

Go to the `Grad-CAM` directory
```
cd Grad-CAM
```
Run the analysis.
```
python feature_analysis_total.py
python feature_analysis_total_Arg.py
python feature_analysis_total_Ser.py
```
And finally, the plots. You'll find them in the `figures` directory as well.
```
python feature_plot_ref2015_heatmap.py
python feature_plot_total_211.py
python feature_plot_total_415.py
python feature_plot_total_Arg_heatmap.py
python feature_plot_total_Ser_heatmap.py
python feature_plot_total.py
```

## Making Predictions

You can easy predict the OI lethality of single point mutatiosn by running `predicter.py`. You will need to input the position and mutation type. For example:
```
python predictor.py -p 247 -m Ser -c a1
```
This will output:
```
Predict result: Lethal
```

## Citing
