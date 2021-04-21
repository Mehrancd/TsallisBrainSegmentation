# SlicerBS
A Slicer module for brain extraction and segmentation

help:

1- Before brain labeling, using check box optimize the brain extract param (higher value, smaller brain mask)
![Alt text](Screenshot1.jpg?raw=true "Using SlicerBS for brain extraction")

2- uncheck the box and set iteration as 1 to provide brain label
![Alt text](Screenshot2.jpg?raw=true "Using SlicerBS for brain label")

3-alpha, beta and gamma are the weights of decision for atlas, intensity and Tsallis approches in Markov Random field 

4-Iteration is for Markov Random field calculation for all voxel neighbors, increasing might take time to finish (6 min for 1 it)

for more details about the algorithm we refer you to https://doi.org/10.1016/j.mri.2019.11.002 https://www.youtube.com/watch?v=gH4dkXm3B2E

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
