import ami.graph_nodes as gn
from ami.flowchart.library.common import CtrlNode
from amitypes import Array1d, Array2d
import numpy as np
import scipy.ndimage as ndimage

# class for keeping track of photons in a droplet
class Photon:
    # initialization (photon gets placed in corner of image)
    def __init__(self,photonSize,photonCount):
        # photonSize in pixels (gaussian shape)
        # photonCount is aduPerPhoton
        self.posX = 0
        self.posY = 0
        self.photonSize = photonSize
        self.photonCount = photonCount

    # update photon position (pixel units)
    def set_pos(self,x,y):
        self.posX = x
        self.posY = y

    # add photon to image, in counts
    def photonModel(self, xIm, yIm):
        # model charge cloud as gaussian
        model = np.exp(-(np.abs(xIm-self.posX)**2 + np.abs(yIm-self.posY)**2) /
                         (2*self.photonSize**2 ) )
        # Make sure it has the correct number of counts
        model = model * self.photonCount / np.sum(model)
        return model

    # add photon to droplet guess, in counts
    def add_photon(self, image, xIm, yIm):
        return image + self.photonModel(xIm, yIm)

    # subtract photon from image, in counts
    def subtract_photon(self, image, xIm, yIm):
        return image - self.photonModel(xIm, yIm)
    
    # update photon map (counting photons, position rounded to nearest pixel)
    def update_map(self, photonMap):
        if round(self.posY) < 0: return
        elif round(self.posY) > 89: return # FIXME: hard-coded Andor size
        elif round(self.posX) < 0: return
        elif round(self.posX) > 89: return # FIXME: hard-coded Andor size

        photonMap[int(round(self.posY)),int(round(self.posX))] += 1

        return

# droplet class for characterizing photon clusters in each image
class Droplet_v1:
    # initialize droplet, based on droplet in Andor image
    def __init__(self, image, label_im, xIm, yIm, 
                 aduPerPhoton, minADU, maxPhotons, photonSize=0.5):
        """
        image: 2D detector image (ADU)
        label_im: 2D image of isolated droplet labels
        xIm: image coordinates
        yIm: image coordinates
        aduPerPhoton: ADUs read out by detector for 1 photon
        minADU: minimum ADU required to register 1 photon
        maxPhotons: maximum number of photons per droplet allowed
        photonSize: size of photon
        """
        # image coordinates
        self.image = image
        self.label_im = label_im
        self.xIm = xIm
        self.yIm = yIm
        self.aduPerPhoton = aduPerPhoton
        self.minADU = minADU
        self.maxPhotons = maxPhotons
        self.photon = Photon(photonSize,aduPerPhoton)

    def find(self, ind):
        """
        ind: droplet index
        returns
        photonMap: 2D image of photons for a droplet at ind
        aduCount: sum of ADUs for a droplet at ind (ADU)
        numPixels: number of pixels for a droplet at ind (pixels)
        """
        # select current droplet
        mask = self.label_im == ind
        # calculate number of pixels in droplet
        numPixels = mask.sum()
        
        # droplet image
        drop = self.image*mask

        # initialize photon map
        photonMap = np.zeros_like(drop,dtype='int')

        # calculate number of ADU in droplet
        aduCount = drop.sum()

        # calculate number of photons in droplet
        numPhotons = int(np.round(aduCount/self.aduPerPhoton))

        if aduCount < self.minADU:
            numPhotons = 0
        elif numPhotons > self.maxPhotons:
            numPhotons = 0

        for i in range(numPhotons):
            # find current peak in droplet
            xPos = np.argmax(np.amax(drop,axis=0))
            yPos = np.argmax(np.amax(drop,axis=1))

            self.photon.set_pos(xPos,yPos)
            # subtract this photon from droplet
            self.photon.subtract_photon(drop, self.xIm, self.yIm)

            # update the guess for the droplet
            photonMap[yPos,xPos] += 1

        return photonMap, aduCount, numPixels

class GreedyGuess_v1:
    def __init__(self, imgShape, threshold, aduPerPhoton, 
                 minADU, maxPhotons, photonSize=0.5):
        """
        imgShape: 2D image pixel dimensions (rows,cols)
        threshold: Noise threshold that is zeroed out (ADUs)
        aduPerPhoton: ADUs read out by detector for 1 photon
        minADU: minimum ADU required to register 1 photon
        maxPhotons: maximum number of photons per droplet allowed
        photonSize: size of photon
        
        returns:
        photonMap: 2D image of detected photons (photons)
        aduCount: list of ADUs found per island
        pixelCount: list of pixels found per island
        """
        self.rows, self.cols = imgShape
        self.threshold = threshold
        self.xIm, self.yIm = np.meshgrid(np.linspace(0,self.rows-1,self.rows),
                                         np.linspace(0,self.cols-1,self.cols))
        self.aduPerPhoton = aduPerPhoton 
        self.minADU = minADU
        self.maxPhotons = maxPhotons
        self.photonSize = photonSize
        
    def findPhotons(self, img):
        mask = img > 0
        img = img * mask # apply mask
        label_im, nb_labels = ndimage.label(mask) 
        photonMap = np.zeros_like(img,dtype='int')
        aduCount = []
        pixelCount = []
        
        drop = Droplet_v1(img, label_im, self.xIm, self.yIm, 
                          self.aduPerPhoton, self.minADU, self.maxPhotons, self.photonSize)

        for i in range(nb_labels):
            _photonMap, _aduCount, _pixelCount = drop.find(i+1)
            photonMap += _photonMap
            aduCount.append( _aduCount )
            pixelCount.append( _pixelCount )
        
        return photonMap#, aduCount, pixelCount

class GreedyGuess(CtrlNode):                                                                                      
                                                                                                                     
    """                                                                                                            
    Projection projects a 2d array along the selected axis.                                                        
    """                                                                                                            
                                                                                                                     
    nodeName = "GreedyGuess"                                                                                      
    uiTemplate = [('aduPerPhoton', 'intSpin', {'value': 300, 'min': 0}),
                  ('minADU', 'intSpin', {'value': 200, 'min': 0}),
                  ('maxPhotons', 'intSpin', {'value': 30, 'min': 0})]                                           
                                                                                                                     
    def __init__(self, name):                                                                                      
        super().__init__(name, terminals={'In': {'io': 'in', 'ttype': Array2d},                                    
                                          'Out': {'io': 'out', 'ttype': Array2d}})                                 
                                                                                                                     
    def to_operation(self, inputs, conditions={}):                                                                 
        outputs = self.output_vars()                                                                               
        aduPerPhoton = self.values['aduPerPhoton']                                                                                 
        minADU = self.values['minADU']
        maxPhotons = self.values['maxPhotons']
        # FIXME: GreedyGuess image size is hard-coded 90x90 below
        self.gg = GreedyGuess_v1((90,90), threshold=20,
                            aduPerPhoton=aduPerPhoton, minADU=minADU, maxPhotons=maxPhotons, photonSize=0.5)
                                                                                                                                                                                              
        node = gn.Map(name=self.name()+"_operation",                                                               
                      condition_needs=conditions, inputs=inputs, outputs=outputs,                                  
                      func=self.gg.findPhotons, parent=self.name())                                                               
        return node
