# SlicerBS
A Slicer module for brain extraction and segmentation

help:

1- Before brain labeling, using check box optimize the brain extract param (higher value, smaller brain mask)

2- uncheck the box and set iteration as 1 to provide brain label

3-alpha, beta and gamma are the weights of decision for atlas, intensity and Tsallis approches in Markov Random field 

4-Iteration is for Markov Random field calculation for all voxel neighbors, increasing might take time to finish (6 min for 1 it)

for more details about the algorithm we refer you to https://doi.org/10.1016/j.mri.2019.11.002

also if you used SlicerBS in your research please cite this paper:

Mehran Azimbagirad, Fabrício H. Simozo, Antonio C.S. Senra Filho, Luiz O. Murta Junior,
Tsallis-Entropy Segmentation through MRF and Alzheimer anatomic reference for Brain Magnetic Resonance Parcellation,
Magnetic Resonance Imaging,
Volume 65,
2020,
Pages 136-145,
ISSN 0730-725X,
https://doi.org/10.1016/j.mri.2019.11.002.
(https://www.sciencedirect.com/science/article/pii/S0730725X19303558)
Abstract: Quantifying the intracranial tissue volume changes in magnetic resonance imaging (MRI) assists specialists to analyze the effects of natural or pathological changes. Since these changes can be subtle, the accuracy of the automatic compartmentalization method is always criticized by specialists. We propose and then evaluate an automatic segmentation method based on modified q-entropy (Mqe) through a modified Markov Random Field (MMRF) enhanced by Alzheimer anatomic reference (AAR) to provide a high accuracy brain tissues parcellation approach (Mqe-MMRF). We underwent two strategies to evaluate Mqe-MMRF; a simulation of different levels of noise and non-uniformity effect on MRI data (7 subjects) and a set of twenty MRI data available from MRBrainS13 as patient brain tissue segmentation challenge. We accessed eleven quality metrics compared to reference tissues delineations to evaluate Mqe-MMRF. MRI segmentation scores decreased by only 4.6% on quality metrics after noise and non-uniformity simulations of 40% and 9%, respectively. We found significant mean improvements in the metrics of the five training subjects, for whole-brain 0.86%, White Matter 3.20%, Gray Matter 3.99%, and Cerebrospinal Fluid 4.16% (p-values < 0.02) when Mqe-MMRF compared to the other reference methods. We also processed the Mqe-MMRF on 15 evaluation subjects group from MRBrainS13 online challenge, and the results held a higher rank than the reference tools; FreeSurfer, SPM, and FSL. Since the proposed method improved the precision of brain segmentation, specifically, for GM, and thus one can use it in quantitative and morphological brain studies.
Keywords: Magnetic resonance imaging; Brain segmentation methods; Markov Random Fields; Tsallis entropy; Atlas segmentation
