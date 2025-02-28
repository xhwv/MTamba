# MTamba
MTamba is a multi-task deep learning network designed for glioma segmentation, IDH genotyping, and tumor grading. We propose a novel architecture that integrates various MRI modalities to handle the complexity and heterogeneity of gliomas. The network is equipped with modules to explore the mismatch between T2 and FLAIR images, as well as multi-modal MRI features, for enhanced tumor diagnosis and grading.

Overview
MTamba aims to solve key challenges in glioma diagnosis:

Glioma Segmentation: Identifying and segmenting tumor regions in multi-modal MRI scans.
IDH Genotyping: Predicting Isocitrate Dehydrogenase (IDH) mutation status, a crucial biomarker for glioma prognosis.
Tumor Grading: Classifying tumor grades, a key factor in prognosis.
Our model design includes the following components:

Tetra-Oriented Mamba: Performs global information interaction from different orientations in MRIs for segmentation.
T2-FLAIR Mismatch Feature Extraction: Explores mismatch features between T2 and FLAIR images at different depths.
Channel-Space Siamese Mamba Fusion: Fuses T2-FLAIR mismatch features with multi-modal MRI features for improved diagnosis.
Uncertainty Loss Optimization: Jointly optimizes glioma segmentation, IDH genotyping, and grading with an uncertainty loss.
We validate MTamba on the UCSF-PDGM and BraTS2020 datasets, and our experimental results demonstrate that MTamba outperforms existing multi-task learning methods in both accuracy and robustness.

Datasets
UCSF-PDGM Dataset
Access the UCSF-PDGM dataset at: UCSF-PDGM Dataset.

BraTS2020 Dataset
Access the BraTS2020 dataset at: BraTS2020 Dataset.

Requirements
Python 3.x
PyTorch 1.x
NumPy
scikit-learn

Usage
Training
Prepare the dataset (UCSF-PDGM or BraTS2020) and place it in the appropriate directory.

To start training, run the following command:

python train.py --dataset ucsf-pdgm --batch-size 2

The model will save checkpoints at the specified path, which can be used for validation or further analysis.


