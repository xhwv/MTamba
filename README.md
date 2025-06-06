# MTamba

MTamba is a multi-task deep learning network designed for glioma segmentation, IDH genotyping, and tumor grading. We propose a novel architecture that integrates various MRI modalities to handle the complexity and heterogeneity of gliomas. The network is equipped with modules to explore the mismatch between T2 and FLAIR images, as well as multi-modal MRI features, for enhanced tumor diagnosis and grading.

**Overview**
MTamba aims to solve key challenges in glioma diagnosis:

**Glioma Segmentation:** Identifying and segmenting tumor regions in multi-modal MRI scans.

**IDH Genotyping:** Predicting Isocitrate Dehydrogenase (IDH) mutation status, a crucial biomarker for glioma prognosis.

**Tumor Grading:** Classifying tumor grades, a key factor in prognosis.

Our model design includes the following components:

Tetra-Oriented Mamba: Performs global information interaction from different orientations in MRIs for segmentation.

T2-FLAIR Mismatch Feature Extraction: Explores mismatch features between T2 and FLAIR images at different depths.

Channel-Space Siamese Mamba Fusion: Fuses T2-FLAIR mismatch features with multi-modal MRI features for improved diagnosis.

Uncertainty Loss Optimization: Jointly optimizes glioma segmentation, IDH genotyping, and grading with an uncertainty loss.

We validate MTamba on the UCSF-PDGM and BraTS2020 datasets, and our experimental results demonstrate that MTamba outperforms existing multi-task learning methods.

**Datasets**

Access the UCSF-PDGM dataset at: [UCSF-PDGM Dataset.](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/.)

BraTS2020 Dataset
Access the BraTS2020 dataset at: https://www.med.upenn.edu/cbica/brats2020/data.html 

**The visualization of SEAM**
We explored T2 and FLAIR images from UCSF-PDGM 188. We visualized both the direct T2 minus FLAIR (w/o SEAM) and the SEAM-enhanced images. The feature value ranges in the visualizations of both w/o SEAM and SEAM are kept consistent. We observed that using SEAM allows for better exploration of tumor-related information.
![e62c17a60ccfc8e7eb8f4a92576ecad](https://github.com/user-attachments/assets/648e196f-938e-4030-9f0c-cf3ea40a43e7)




Requirements

torch>=1.10.0
monai>=0.9.0
torchvision>=0.11.0
scipy>=1.7.0
numpy>=1.21.0
pickle5>=0.0.11


Usage

Training

Prepare the dataset (UCSF-PDGM or BraTS2020) and place it in the appropriate directory.

To start training, run the following command:

python train.py --dataset ucsf-pdgm --batch-size 2

The model will save checkpoints at the specified path, which can be used for validation or further analysis.


