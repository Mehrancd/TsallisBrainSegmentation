# Tsallis Brain Segmentation
A Slicer module for brain extraction and segmentation

Help:

1- Before brain labeling, you can use the first box to create a mask for the brain (brain extractor parameter for higher value, smaller brain mask will be created using beepbrain module)
![Alt text](Screenshot1.jpg?raw=true "Using SlicerBS for brain extraction")

2- For brain segmentation, a head image, a mask and a lable map for export the result is needed.
![Alt text](Screenshot2.jpg?raw=true "Using SlicerBS for brain label")

- q is the parameter for Tsallis entropy calculation (Sq).
- alpha is the parameter showing the weight of Tsallis entropy-Markov decision 
- beta is the parameter showing the weight of image histogram intensity decision
- gamma  is the parameter showing the weight of Atlas based decision for alzheimer brain images (only for such patient is effective)
-Iteration is for Markov Random field calculation for all voxel neighbors, increasing might take time to finish (6 min to 1 hour)
- If the results are not satisfying, check if the eyes are wrongly detected or not. They should be remove from the mask.

for more details about the algorithm we refer you to https://doi.org/10.1016/j.mri.2019.11.002 https://www.youtube.com/watch?v=gH4dkXm3B2E

also if you used Tsallis Brain Segmentation in your research please cite this paper:

Mehran Azimbagirad, Fabrício H. Simozo, Antonio C.S. Senra Filho, Luiz O. Murta Junior,
Tsallis-Entropy Segmentation through MRF and Alzheimer anatomic reference for Brain Magnetic Resonance Parcellation,
Magnetic Resonance Imaging,
Volume 65,
2020,
Pages 136-145,
ISSN 0730-725X,
https://doi.org/10.1016/j.mri.2019.11.002.
(https://www.sciencedirect.com/science/article/pii/S0730725X19303558)
