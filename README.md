# HLTPGO: Harnessing Long-Tail Phenomenon for Accurate Protein Function Prediction

<div align="center">
  <img src="Fig/Overview.png" width="100%"/>
</div>
we propose the HLTPGO method, which incorporates a long-tail optimization ensemble strategy. HLTPGO employs the base model HLTPGO Sig, introduces a global-local label graph module, and utilizes a multi-granularity focal loss function to enhance long-tail functional information, mitigate the long-tail phenomenon, and improve overall prediction accuracy.

## Installation

```bash
# clone the repo
git clone https://github.com/Cao-Labs/HLTPGO.git
cd HLTPGO
```
We use python 3.9.19 and pytorch 1.13.0. We recommend using conda to install the dependencies:
```bash
conda env create -f environment.yml
```
You also need to install the relative packages to run ESM-1b protein language model. \
Please see [facebookresearch/esm](https://github.com/facebookresearch/esm#getting-started-with-this-repo-) for details. \

## Usage

### Model Training:
```bash
python train.py --train_part ${base_model_nane} --batch_size ${batch_size} --epochs ${max_epoch} --sub_function ${sub-functional ontologies} --AF2model ${enhanced} --config-file $ {config}
```
train.py is used to train the basic model. where `--train_part` is the type of the base model for training， including {All, Head, Tail}; `--batch_size` is the size of each batch; `--epochs` is the maximum number of iterations for training, `--sub_function` are the specific functional categories trained, including {mf, bp, cc}, `--AF2model` is whether to use the AF2 dataset to enhance the training data(The default value is False); `--config-file` is the path to the model configuration file. 

### Model Test:
```bash
python test.py --train_part ${base_model_nane} --batch_size ${batch_size} --dataset  ${test-dataset} --sub_function ${sub-functional ontologies} --AF2model ${enhanced} --config-file $ {config}
```
test.py is used to test the prediction performance of the base model. where `--train_part` is the type of the base model， including {All, Head, Tail}; `--batch_size` is the size of each batch; `--dataset` is the test data set, where AF2test is the AF2 test set and test is the PDB test set; `--sub_function` are the specific functional categories tested, including {mf, bp, cc}; `--AF2model` is True, indicating that the test set is the AF2 test set(The default value is False, indicating the PDB test set); `--config-file` is the path to the model configuration file. 

### Ensemble:
```bash
python Ensemble.py --ontology ${sub-functional ontologies}
```
Ensemble.py is used to integrate the training results of the basic model. where `--ontology` are the specific functional categories tested, including {mf, bp, cc}. 

### Protein Function Prediction:
```bash
python Predictor.py  --sub_function ${sub-functional ontologies} --config-file $ {config} --pdb ${pdb-fila} --prob $ {threshold}
```
Predictor.py is used to predict protein function. where `--sub_function` is the type of function to be predicted, including {mf, bp, cc}; `--config-file` is the path to the model configuration file; `--pdb` is the path to the protein PDB file; `--prob` is the threshold for feature prediction.
