import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np
import SimpleITK as sitk
import time

class TsallisBrainSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TsallisBrainSegmentation"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Mehran Azimbagirad, Fabrício H Simozo, Antonio CS Senra Filho, Luiz O Murta Junior (CSIM Lab)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
1-Brain extraction parameter is for deep brain extraction, higer value is smaller volume
2-q parameter is for Tsallis entropy estimation, q=1 is shannon entropy
3-alpha, beta and gamma are the weights of decision for atlas, intensity and Tsallis approches in Markov Random field
4-Iteration is for Markov Random field calculation for all voxel neighbors, increasing might take time to finish
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
Tsallis-Entropy Segmentation through MRF and Alzheimer anatomic reference for Brain Magnetic Resonance Parcellation
https://doi.org/10.1016/j.mri.2019.11.002
https://pypi.org/project/deepbrain/
"""
    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)
# Register sample data sets in Sample Data module
def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.
  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # BS1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='TsallisBrainSegmentation',
    sampleName='TsallisBrainSegmentation1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'TsallisBrainSegmentation.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='TsallisBrainSegmentation1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='TsallisBrainSegmentation1'
  )

  # BS2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='TsallisBrainSegmentation',
    sampleName='TsallisBrainSegmentation2',
    thumbnailFileName=os.path.join(iconsPath, 'BS2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='TsallisBrainSegmentation2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='TsallisBrainSegmentation2'
  )
# Widget
class TsallisBrainSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/TsallisBrainSegmentation.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = TsallisBrainSegmentationLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector_1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inputSelector_2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inputSelector_3.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector_1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector_2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.bepSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.qParameterctkSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.alphactkSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.betactkSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.gammactkSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.iterationctkSliderWidget.connect("valueChanged(int)", self.updateParameterNodeFromGUI)
    #self.ui.onlyExtractBrainctkCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    #self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    #self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
    #self.ui.applyButton_2.connect('clicked(bool)', self.onApplyButton)
    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    self.removeObservers()
  def enter(self):
    self.initializeParameterNode()
  def exit(self):
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
  def onSceneStartClose(self, caller, event):
    self.setParameterNode(None)
  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.initializeParameterNode()
  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.
    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    #if not self._parameterNode.GetNodeReference("InputVolume1"):
    #  firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    #  if firstVolumeNode:
    #    self._parameterNode.SetNodeReferenceID("InputVolume1", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)
    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector_1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume1"))
    self.ui.inputSelector_2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume2"))
    self.ui.inputSelector_3.setCurrentNode(self._parameterNode.GetNodeReference("InputMask"))
    self.ui.outputSelector_1.setCurrentNode(self._parameterNode.GetNodeReference("OutputMask"))
    self.ui.outputSelector_2.setCurrentNode(self._parameterNode.GetNodeReference("OutputLabel"))
    #self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    self.ui.bepSliderWidget.value = float(self._parameterNode.GetParameter("bep"))
    self.ui.qParameterctkSliderWidget.value = float(self._parameterNode.GetParameter("q"))
    self.ui.alphactkSliderWidget.value = float(self._parameterNode.GetParameter("alpha"))
    self.ui.betactkSliderWidget.value = float(self._parameterNode.GetParameter("beta"))
    self.ui.gammactkSliderWidget.value = float(self._parameterNode.GetParameter("gamma"))
    self.ui.iterationctkSliderWidget.value = int(float(self._parameterNode.GetParameter("iter")))
    #self.ui.onlyExtractBrainctkCheckBox.checked = (self._parameterNode.GetParameter("mask") == "true")
    #self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume1") and self._parameterNode.GetNodeReference("OutputMask") :
      self.ui.applyButton.toolTip = "Create Brain LabelMap"
      self.ui.applyButton.enabled = True
    elif self._parameterNode.GetNodeReference("InputVolume2") and self._parameterNode.GetNodeReference("InputMask") and self._parameterNode.GetNodeReference("OutputLabel"):
      self.ui.applyButton.toolTip = "Create Brain LabelMap"
      self.ui.applyButton.enabled = True
        
    else:
      self.ui.applyButton.toolTip = "Select head volume and mask and brain Labelmap"
      self.ui.applyButton.enabled = False
    # push bottons 2
#    if self._parameterNode.GetNodeReference("InputVolume2") and self._parameterNode.GetNodeReference("OutputLabel")and self._parameterNode.GetNodeReference("InputMask"):
#      self.ui.applyButton_2.toolTip = "Create Brain LabelMap"
#      self.ui.applyButton_2.enabled = True
#    else:
#      self.ui.applyButton_2.toolTip = "Select head volume and mask"
#      self.ui.applyButton_2.enabled = False
    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolume1", self.ui.inputSelector_1.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.inputSelector_2.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputMask", self.ui.inputSelector_3.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputMask", self.ui.outputSelector_1.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputLabel", self.ui.outputSelector_2.currentNodeID)
    self._parameterNode.SetParameter("bep", str(self.ui.bepSliderWidget.value))
    self._parameterNode.SetParameter("q", str(self.ui.qParameterctkSliderWidget.value))
    self._parameterNode.SetParameter("alpha", str(self.ui.alphactkSliderWidget.value))
    self._parameterNode.SetParameter("beta", str(self.ui.betactkSliderWidget.value))
    self._parameterNode.SetParameter("gamma", str(self.ui.gammactkSliderWidget.value))
    self._parameterNode.SetParameter("iter", str(self.ui.iterationctkSliderWidget.value))
    #self._parameterNode.SetParameter("mask", "true" if self.ui.onlyExtractBrainctkCheckBox.checked else "false")
    #self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
        import tensorflow
        import scipy
        import skimage  
    except ModuleNotFoundError as e:
        if slicer.util.confirmOkCancelDisplay("This module requires 'tensorflow' Python package. Click OK to install (it takes several minutes)."):
            slicer.util.pip_install("tensorflow")
            slicer.util.pip_install("scipy")
            slicer.util.pip_install("scikit-image")
            #import tensorflow
            import scipy
            import skimage 

    try:
        
      # Compute output
      self.logic.process(self.ui.inputSelector_1.currentNode(),self.ui.inputSelector_2.currentNode(),self.ui.inputSelector_3.currentNode(), self.ui.outputSelector_1.currentNode(),self.ui.outputSelector_2.currentNode(),self.ui.bepSliderWidget.value, self.ui.qParameterctkSliderWidget.value, self.ui.alphactkSliderWidget.value, self.ui.betactkSliderWidget.value, self.ui.gammactkSliderWidget.value, self.ui.iterationctkSliderWidget.value, self._parameterNode.GetNodeReference("InputMask"))

      # Compute inverted output (if needed)
      # if self.ui.invertedOutputSelector.currentNode():
      #   # If additional output volume is selected then result with inverted threshold is written there
      #   self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
      #     self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()



#

class TsallisBrainSegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("bep"):
      parameterNode.SetParameter("bep", "0.999")
    if not parameterNode.GetParameter("q"):
      parameterNode.SetParameter("q", "-0.64")
    if not parameterNode.GetParameter("alpha"):
      parameterNode.SetParameter("alpha", "0.5")
    if not parameterNode.GetParameter("beta"):
      parameterNode.SetParameter("beta", "0.5")
    if not parameterNode.GetParameter("gamma"):
      parameterNode.SetParameter("gamma", "0.0")
    if not parameterNode.GetParameter("iter"):
      parameterNode.SetParameter("iter", "1")
    # if not parameterNode.GetParameter("Invert"):
    #   parameterNode.SetParameter("Invert", "false")

  def process(self, inputVolume1,inputVolume2,inputVolume3, outputVolume1, outputVolume2, bep, q, alpha, beta, gamma, Iteration , showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """
    from scipy.stats import norm
        
    if not(inputVolume1 and outputVolume1) and not(inputVolume2 and inputVolume3 and outputVolume2):
      print('mask extractor logic:',not(inputVolume1 and outputVolume1))
      raise ValueError("Input or output volume/label is/are invalid")

    startTime = time.time()
    logging.info('Processing started')
    progressbar = slicer.util.createProgressDialog(autoClose=True)
    progressbar.value = 5
    progressbar.labelText = "processing started ..."
    #------------------- Deep brain ------------------------------------------+
    if (inputVolume1 and outputVolume1) and not(inputVolume2 and inputVolume3 and outputVolume2):
        
        from Resources import Extractor
        print('1-Brain extraction is started using Deep Brain extractor...')
        ext = Extractor()
        t1Array =slicer.util.array(inputVolume1.GetID())
        prob = ext.run(t1Array) 
        # mask can be obtained as: where the best parameters was found for 0.125
        prob[prob<bep]=0
        prob[prob>=bep]=1
        mask=prob
        volumesLogic = slicer.modules.volumes.logic()
        outputVolume = volumesLogic.CreateLabelVolumeFromVolume(slicer.mrmlScene, outputVolume1,inputVolume1 )
        ijkToRas = vtk.vtkMatrix4x4()
        outputVolume1.GetIJKToRASMatrix(ijkToRas)
        imageData=outputVolume.GetImageData()
        extent = imageData.GetExtent()
        for k in range(extent[4], extent[5]+1):
            for j in range(extent[2], extent[3]+1):
                for i in range(extent[0], extent[1]+1):
                    imageData.SetScalarComponentFromDouble(i,j,k,0,mask[k,j,i])
        imageData.Modified()
        slicer.util.setSliceViewerLayers(background=slicer.util.getNode(inputVolume1.GetName()), foreground='keep-current', label=slicer.util.getNode(outputVolume1.GetName()), foregroundOpacity=None, labelOpacity=0.5, fit=False) #, rotateToVolumePlane=False
        progressbar.value = 100
    else:
        t1Array =slicer.util.array(inputVolume2.GetID())
        mask=slicer.util.array(inputVolume3.GetID())
        mask[mask!=0]=1
        progressbar.value = 15
        brain=np.multiply(t1Array,mask)
        brain=np.absolute(brain)
        MaxIn=brain.max()
        MinIn=brain.min()
        brain=(brain*255.0)/MaxIn
        MaxIn=brain.max()
        MinIn=brain.min()
        brain=np.ceil(brain.astype(np.float32))
        print('2-Image labeling is started using Tsallis entropy...')
        label_map=Mqe(brain,q)
        print('Image labeling done.')
        print('3-Label correcting is started using MRF and AAR...')
        Atlas = sitk.ReadImage(os.path.dirname(os.path.realpath(__file__))+'/Resources/Atlas.nii.gz')
        Atlas_label = sitk.ReadImage(os.path.dirname(os.path.realpath(__file__))+'/Resources/Atlas_label.nii.gz')
        brain_image=sitk.GetImageFromArray(brain)
        brain_image.SetOrigin(Atlas.GetOrigin())
        dirs = np.zeros([3,3])
        inputVolume3.GetIJKToRASDirections(dirs)
        dirs=np.multiply(dirs,np.array([[-1,-1,-1],[-1,-1,-1],[1,1,1]]))
        direction=np.reshape(dirs,(1,9)).tolist()
        brain_image.SetDirection(direction[0])
        brain_image.SetSpacing(inputVolume3.GetSpacing())
        Atlas_registered, label_registered =Register_7DOF(brain_image,Atlas,Atlas_label)
        Atlas_registered = sitk.GetArrayFromImage(Atlas_registered)
        muA,sigmaA=label_statistic(Atlas_registered,sitk.GetArrayFromImage(label_registered))
        numberOfIterations=int(Iteration)
        n_weights=np.array([[1,1,1,1,2,1,1,1,1,1,2,1,2,0,2,1,2,1,1,1,1,1,2,1,1,1,1]]).transpose()
        for it in range(numberOfIterations):
            progressbar.value = 15+(it/numberOfIterations)*85-1
            muI,sigmaI=label_statistic(brain,label_map)
            for k in range(brain.shape[0]): 
                if brain[k,:,:].max()==0:
                    continue
                for i in range(brain.shape[1]): 
                    for j in range(brain.shape[2]): 
                        neighbors=GetNeighbors(brain,[k,i,j])
                        target=brain[k,i,j]
                        if target==0 and neighbors[np.nonzero(neighbors)].size<10:
                            continue
                        elif target==0:
                            target=neighbors.mean()
                        n_labels=GetNeighbors(label_map,[k,i,j])
                        C1l=np.copy(n_labels)
                        C1l[C1l!=1]=0
                        C2l=np.copy(n_labels)
                        C2l[C2l!=2]=0
                        C3l=np.copy(n_labels)
                        C3l[C3l!=3]=0
                        C1=np.multiply(neighbors,C1l)
                        C2=np.multiply(neighbors,C2l/2)
                        C3=np.multiply(neighbors,C3l/3)
                        mmrf=np.array([MRF(C1,C2,C3,n_weights,target)])
                        Image_weights=norm.pdf(target,loc=muI,scale=sigmaI).transpose()
                        Image_weights[0,0]=Image_weights.min()
                        Atlas_weights=norm.pdf(Atlas_registered[k,i,j],loc=muA,scale=sigmaA).transpose()
                        Atlas_weights[0,0]=Atlas_weights.min()
                        Ut=alpha*mmrf/mmrf.max()-beta*Image_weights/Image_weights.max()-gamma*Atlas_weights/Atlas_weights.max()
                        new_label=np.argmin(Ut)
                        label_map[k,i,j]=new_label
                print(np.floor((k/brain.shape[0])*100),'%')
        print('100.0%')
        progressbar.value =100
        volumesLogic = slicer.modules.volumes.logic()
        outputVolume2 = volumesLogic.CreateLabelVolumeFromVolume(slicer.mrmlScene, outputVolume2,inputVolume2 )
        ijkToRas = vtk.vtkMatrix4x4()
        outputVolume2.GetIJKToRASMatrix(ijkToRas)
        imageData=outputVolume2.GetImageData()
        extent = imageData.GetExtent()
        for k in range(extent[4], extent[5]+1):
            for j in range(extent[2], extent[3]+1):
                for i in range(extent[0], extent[1]+1):
                    imageData.SetScalarComponentFromDouble(i,j,k,0,label_map[k,j,i])
        imageData.Modified()
        slicer.util.setSliceViewerLayers(background=slicer.util.getNode(inputVolume2.GetName()), foreground='keep-current', label=slicer.util.getNode(outputVolume2.GetName()), foregroundOpacity=None, labelOpacity=0.5, fit=False)
    

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# BSTest
#

class TsallisBrainSegmentationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()
  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_TsallisBrainSegmentation1()

  def test_TsallisBrainSegmentation1(self):
  
    

    self.delayDisplay('Test passed')
#*****************************************************************************
#*****************************************************************************
#*****************************************************************************
#*****************************************************************************
from scipy.stats import norm
def Mqe(brain,q): 
    #Histogram of the image
    Hist, bin_edges=np.histogram(brain,bins=range(0,257))
    #Hist=plt.hist(brain.ravel(), bins=range(1,int(MaxIn)+1))
    Hist[0]=0;
    norm_Hist=Hist/Hist.sum()  
    for i in range(1,Hist.size):
        if norm_Hist[i]>2.0e-5:
            first_bin=i
            break
    for i in range(Hist.size-1,1,-1):
        if norm_Hist[i]>2.0e-5:
            last_bin=i
            break
    Num_freq=np.sum(Hist)
# Mqe segmentation:calculating q-entropy to find the maximum entropy or t1 and t2 for thresholding
    maxim=-1.0e300
    for t1 in range(first_bin+2,last_bin-2):
        for t2 in range(t1+1,last_bin):
            num_freq1=np.sum(Hist[first_bin:t1+1])
            num_freq2=np.sum(Hist[t1+1:t2+1])
            num_freq3=np.sum(Hist[t2+1:last_bin]) 
            if num_freq1==0.0 or num_freq2==0.0 or num_freq3==0.0:
                continue
            weights1=Hist[first_bin:t1+1]/num_freq1
            weights2=Hist[t1+1:t2+1]/num_freq2
            weights3=Hist[t2+1:last_bin]/num_freq3                                   
            mu=np.array([np.average(range(first_bin,t1+1),weights=weights1), 
                     np.average(range(t1+1,t2+1),weights=weights2),
                     np.average(range(t2+1,last_bin),weights=weights3)])
            sigma=np.array([np.average((range(first_bin,t1+1)-mu[0])**2,weights=weights1), 
                     np.average((range(t1+1,t2+1)-mu[1])**2,weights=weights2),
                     np.average((range(t2+1,last_bin)-mu[2])**2,weights=weights3)])
            for i in range(3):
                if sigma[i]<1:
                    sigma[i]+=1
            pro=np.array([ num_freq1, num_freq2, num_freq3]/Num_freq)
            Xi=range(0,255)
            norm.pdf(3,loc=5,scale=1)
            GMM=pro[0]*norm.pdf(Xi,loc=mu[0],scale=sigma[0])+pro[1]*norm.pdf(Xi,loc=mu[1],scale=sigma[1])+pro[2]*norm.pdf(Xi,loc=mu[2],scale=sigma[2])
            SA=np.sum(np.power(GMM[first_bin:t1+1]/np.sum(GMM[first_bin:t1+1]),q))
            SA=(1-SA)/(q-1)
            SB=np.sum(np.power(GMM[t1+1:t2+1]/np.sum(GMM[t1+1:t2+1]),q))
            SB=(1-SB)/(q-1)
            SC=np.sum(np.power(GMM[t2+1:last_bin]/np.sum(GMM[t2+1:last_bin]),q))
            SC=(1-SC)/(q-1)
            topt=SA+SB+SC+(1-q)*(SA*SB+SA*SC+SB*SC)+(1-q)*(1-q)*SA*SB*SC;
            if(topt>=maxim):
                maxim=topt;
                T1=t1;
                T2=t2;          
    #print(T1,T2)
    label_map=np.zeros(brain.shape)
    label_map[(brain>0) & (brain<T1)]=1
    label_map[(brain>=T1) & (brain<T2)]=2
    label_map[brain>=T2]=3
    return label_map
#*****************************************************************************
def Register_7DOF(fixed,moving,label):
    samplingPercentage = 0.002
    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMeanSquares()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(0.05,.001,1500,0.5)
    tx=sitk.CenteredTransformInitializer(fixed, moving,sitk.ScaleVersor3DTransform())
    R.SetInitialTransform(tx,inPlace=True)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetNumberOfThreads(16)
    outTx = R.Execute(fixed, moving)
    print("-------")
    #print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))
    registered_image=sitk.Resample(moving, fixed, outTx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    registered_label=sitk.Resample(label, fixed, outTx, sitk.sitkLinear, 0.0, label.GetPixelID())
    #type(registered_image)
    return registered_image, registered_label
#*****************************************************************************
def label_statistic(Image,label):
    muI=np.zeros([4,1])
    sigmaI=np.ones([4,1])
    muI[1]=Image[(Image>0) & (label==1)].mean()
    muI[2]=Image[(Image>0) & (label==2)].mean()
    muI[3]=Image[(Image>0) & (label==3)].mean()
    sigmaI[1]=np.sqrt(Image[(Image>0) & (label==1)].var())
    sigmaI[2]=np.sqrt(Image[(Image>0) & (label==2)].var())
    sigmaI[3]=np.sqrt(Image[(Image>0) & (label==3)].var())
    return muI,sigmaI
#*****************************************************************************
def GetNeighbors(img,Ind):
    p,r,c=Ind
    neighborhood=np.zeros([27,1])
    try:
        neighborhood[0] = img[p-1, r-1, c-1]
        neighborhood[1] = img[p-1, r,   c-1]
        neighborhood[2] = img[p-1, r+1, c-1]
    
        neighborhood[ 3] = img[p-1, r-1, c]
        neighborhood[ 4] = img[p-1, r,   c]#dist 1
        neighborhood[ 5] = img[p-1, r+1, c]
    
        neighborhood[ 6] = img[p-1, r-1, c+1]
        neighborhood[ 7] = img[p-1, r,   c+1]
        neighborhood[ 8] = img[p-1, r+1, c+1]

        neighborhood[ 9] = img[p, r-1, c-1]
        neighborhood[10] = img[p, r,   c-1]#dist 1
        neighborhood[11] = img[p, r+1, c-1]

        neighborhood[12] = img[p, r-1, c]#dist 1
        neighborhood[13] = 0 #img[p, r,   c] center
        neighborhood[14] = img[p, r+1, c]#dist 1

        neighborhood[15] = img[p, r-1, c+1]
        neighborhood[16] = img[p, r,   c+1]#dist 1
        neighborhood[17] = img[p, r+1, c+1]

        neighborhood[18] = img[p+1, r-1, c-1]
        neighborhood[19] = img[p+1, r,   c-1]
        neighborhood[20] = img[p+1, r+1, c-1]

        neighborhood[21] = img[p+1, r-1, c]
        neighborhood[22] = img[p+1, r,   c]#dist 1
        neighborhood[23] = img[p+1, r+1, c]

        neighborhood[24] = img[p+1, r-1, c+1]
        neighborhood[25] = img[p+1, r,   c+1]
        neighborhood[26] = img[p+1, r+1, c+1]
    except:
        p,r,c=Ind
    return neighborhood
#*****************************************************************************
def MRF(V1,V2,V3,W,target):
    MRF_weights=np.zeros([4])
    mu=np.zeros([4])
    sigma=np.ones([4])
    mu[1], sigma[1]=weighted_stat_info(V1,W)
    mu[2], sigma[2]=weighted_stat_info(V2,W)
    mu[3], sigma[3]=weighted_stat_info(V3,W)
    V1size=V1[np.nonzero(V1)].size
    V2size=V1[np.nonzero(V2)].size
    V3size=V1[np.nonzero(V3)].size
    if target==0:
        MRF_weights=[1,2,3,4] #[background CSF GM WM] the min is the best
    else:
        MRF_weights[0]=4
        if V1size==0 and V2size==0 and V3size==0: #alone voxel
            MRF_weights=[1,2,3,4]
        elif V1size==0 and V2size==0 and V3size!=0: #Target has one neighbor in WM
            MRF_weights=[4,3,2,1]
        elif V1size==0 and V2size!=0 and V3size==0: #Target has one neighbor in GM
            MRF_weights=[4,2,1,3]
        elif V1size!=0 and V2size==0 and V3size==0: #Target has one neighbor in CSF
            MRF_weights=[4,1,2,3]
        elif V1size==0 and V2size!=0 and V3size!=0: #**********************************Target has two neighbor WM+GM  
            if target>=mu[3]+2*sigma[3] and V3size>2:
                MRF_weights=[4,3,2,1]
            elif target<mu[2]-2*sigma[2]: #decision is not clear maybe in GM or even CSF
                MRF_weights=[4,0,0,0]
            else:
                MRF_weights[1]=3 # is not in CSF
                #print(mu)
                #print(sigma)
                if (mu[2]+2*sigma[2])<(mu[3]-2*sigma[3]) and V3size>2 and V2size>2:
                    T=mu[2]+sigma[2]-((mu[2]+sigma[2])-(mu[3]-sigma[3]))/2 # may need consider
                    MRF_weights[2]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
        elif V1size!=0 and V2size==0 and V3size!=0: #*********************************Target has two neighbor CSF+WM
            if target>mu[3]:
                MRF_weights=[4,2,3,1]
            elif target<(mu[3]-2*sigma[3]) and target>(mu[1]+2*sigma[1]): #decision is not clear in any tissue
                MRF_weights=[4,0,0,0]
            elif target<mu[1]:
                MRF_weights=[4,1,3,2]
            else:
                MRF_weights[2]=3 # is not in GM
                if (mu[1]+sigma[1])<(mu[3]-sigma[3]) and V1size>2 and V3size>2:
                    T=mu[1]+sigma[1]+((mu[3]+sigma[3])-(mu[1]-sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    if (max1/max2)>1:
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[1],scale=sigma[1])
                    else:
                        MRF_weights[1]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])
        elif V1size!=0 and V2size!=0 and V3size==0: #************************************Target has two neighbor CSF+GM  
            if target>mu[2]+2*sigma[2] and V2size>2:#decision is not clear in any tissue
                MRF_weights=[4,0,0,0]
            elif target<mu[1]-2*sigma[1] and V1size>2:
                MRF_weights=[4,1,2,3]
            else:
                MRF_weights[3]=3 # is not in WM
                if (mu[1]+2*sigma[1])<(mu[2]-2*sigma[2]) and V1size>2 and V2size>2:
                    T=mu[1]+sigma[1]+((mu[2]-sigma[2])-(mu[1]+sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[2]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])#that is inverse because min is used
                        MRF_weights[1]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[1],scale=sigma[1]) #that is inverse because finaly the min is used
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
        else:# ******************************************************************************target has three neighbors
            if target>mu[3]+2*sigma[3] and V3size>2:
                MRF_weights=[4,3,2,1]
            elif target<mu[1]-2*sigma[1] and V1size>2:
                MRF_weights=[4,1,2,3]
            #elif target>=mu[1]-2*sigma[1] and target<mu[2]+sigma[2]: #target between CSF and GM
            elif target<mu[2]: #target between CSF and GM
                MRF_weights[3]=3 # is not in WM
                if (mu[1]+sigma[1])<(mu[2]-sigma[2]) and V1size>2 and V2size>2:
                    T=mu[1]+sigma[1]+((mu[2]-sigma[2])-(mu[1]+sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[2]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])#that is inverse because min is used
                        MRF_weights[1]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[1],scale=sigma[1]) #that is inverse because finaly the min is used
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
            else:#target between GM and WM
                MRF_weights[1]=3 # is not in CSF
                if (mu[2]+2*sigma[2])<(mu[3]-2*sigma[3]) and V3size>2 and V2size>2:
                    T=mu[2]+sigma[2]-((mu[2]+sigma[2])-(mu[3]-sigma[3]))/2
                    MRF_weights[2]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
  
    return MRF_weights
#******************************************************************************
def weighted_stat_info(V,W):
    #mu_n[3]=C3[np.nonzero(C3)].mean()*sum(1*(C3!=0))/sum(1*((n_labels==3)*(n_weights)))
    #sigma_n[1]=np.sqrt(C1[np.nonzero(C1)].var()*sum(1*(C1!=0))/sum(1*((n_labels==1)*(n_weights))))
    mu=0.0
    sigma=1
    sum_lw=sum((1*(V!=0))*W)
    V=np.multiply(V,W)
    #print(sum_lw)
    if sum_lw > 0 :
        mu=V[np.nonzero(V)].mean()*sum(1*(V!=0))/sum_lw
        sigma=np.sqrt(V[np.nonzero(V)].var()*sum(1*(V!=0))/sum_lw)
    if sigma<1:
        sigma=1        
    return mu,sigma
