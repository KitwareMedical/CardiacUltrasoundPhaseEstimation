import numpy as np
import scipy.ndimage
import time
import sys
import sklearn.decomposition
import statsmodels.api as sm
import angles

sys.path.insert(0, 'pyLAR')
import core.ialm

def normalizeAngles(angleList, angle_range):
    return np.array([angles.normalize(i, angle_range[0], angle_range[1]) for i in angleList])

class USGatingAndSuperResolution(object):
    
    def __init__(self, 
                 # noise suppression parameters                 
                 medianFilterSize = 3, xyDownsamplingFactor = 1, gammaFactor = 1,
                 # phase estimation parameters
                 similarityMethod = 'PCA', pca_n_components = 0.99, 
                 lowessFrac=0.3, detrend_mode = '/'
                 ):
        
        # noise suppression parameters
        self.medianFilterSize = medianFilterSize        
        self.gammaFactor = gammaFactor
        
        if not (xyDownsamplingFactor > 0 and xyDownsamplingFactor <= 1.0):
            raise ValueError('xyDownsamplingFactor should in (0, 1]')            
        self.xyDownsamplingFactor = xyDownsamplingFactor
        
        # phase estimation parameters
        if similarityMethod not in ['NCORR', 'PCA']:
            raise ValueError("Invalid similarity method. Must be NCORR or PCA")            
        self.similarityMethod = similarityMethod 

        self.pca_n_components = pca_n_components

        if detrend_mode not in ['/', '-']:
            raise ValueError("Invalid detrend mode. Must be '/' or '-'")            
        self.detrend_mode = detrend_mode
        
        if not (lowessFrac > 0 and lowessFrac <= 1.0):
            raise ValueError('lowessFrac should in (0, 1]')            
        self.lowessFrac = lowessFrac
        
    def _denoise(self, imInput):
        
        # smooth to reduce noise if requested
        imInputForLowrank = imInput

        if self.medianFilterSize > 0:
            imInputForLowrank = scipy.ndimage.filters.median_filter(imInput, 
                                                                    (self.medianFilterSize, 
                                                                     self.medianFilterSize, 1))

        # reduce xy size to speed up low-rank + sparse dcomposition if requested 
        if self.xyDownsamplingFactor < 1:    
            imInputForLowrank = scipy.ndimage.interpolation.zoom(imInputForLowrank, 
                                                                 (self.xyDownsamplingFactor, 
                                                                  self.xyDownsamplingFactor, 1))  

        # create a matrix D where each column represents data at one time point
        D = np.reshape(imInputForLowrank, (np.prod(imInputForLowrank.shape[:2]), imInputForLowrank.shape[2]))    

        # perform low-rank plus sparse decomposition on D
        tRPCA = time.time()

        gamma = self.gammaFactor / np.sqrt(np.max(D.shape))

        res = core.ialm.recover(D, gamma)

        D_lowRank = np.array( res[0] )
        D_sparse = np.array( res[1] )

        imD = np.reshape(D, imInputForLowrank.shape)
        imLowRank = np.reshape(D_lowRank, imInputForLowrank.shape)
        imSparse = np.reshape(D_sparse, imInputForLowrank.shape)

        if self.xyDownsamplingFactor < 1:    

            # restore result to original size
            zoomFactor = np.array(imInput.shape).astype('float') / np.array(imInputForLowrank.shape)
            imD = scipy.ndimage.interpolation.zoom(imD, zoomFactor)  
            imLowRank = scipy.ndimage.interpolation.zoom(imLowRank, zoomFactor)  
            imSparse = scipy.ndimage.interpolation.zoom(imSparse, zoomFactor)  

        print 'Low-rank plus sparse decomposition took {} seconds'.format(time.time() - tRPCA)  
        
        # store results
        self.imD_ = imD
        self.imLowRank_ = imLowRank     
        self.imSparse_ = imSparse
        
    def _compute_frame_similarity(self, imAnalyze):
        
        # Compute similarity of data at each time point with all other timepoints
        simMat = np.zeros((imAnalyze.shape[-1], imAnalyze.shape[-1]))

        if self.similarityMethod == 'NCORR':

            print '\nComputing frame similarity matrix using normalized correlation ... ', 

            tSimMat = time.time()

            X = np.reshape(imAnalyze, (np.prod(imAnalyze.shape[:2]), imAnalyze.shape[-1]))
            X = (X - X.mean(0)) / X.std(0)
            simMat = np.dot(X.T, X) / X.shape[0]    

            print 'took {} seconds'.format(time.time() - tSimMat)

        elif self.similarityMethod == 'PCA':  

            # create a matrix where each row represents one frame
            X = np.reshape(imAnalyze, (np.prod(imAnalyze.shape[:2]), imAnalyze.shape[-1])).T

            # perform pca on X
            print 'Reducing dimensionality using PCA ... ', 

            tPCA_Start = time.time()

            pca = sklearn.decomposition.PCA(n_components = self.pca_n_components)

            X_proj = pca.fit_transform(X)

            tPCA_End = time.time()

            numEigenVectors = pca.n_components_

            print 'took {} seconds'.format(tPCA_End - tPCA_Start)
            print '%d eigen vectors were needed to cover %.2f%% of variance' % (numEigenVectors, 
                                                                                self.pca_n_components * 100)

            # Compute similarity of key frame with all the other frames
            print '\nComputing frame similarity matrix as -ve distance in pca-reduced space ... ' 

            simMat = np.zeros((imAnalyze.shape[2], imAnalyze.shape[2]))

            tSimMat = time.time()

            for keyFrameId in range(imAnalyze.shape[2]):

                for fid in range(keyFrameId, imAnalyze.shape[2]):

                    p2pVec = X_proj[fid, :numEigenVectors] - X_proj[keyFrameId, :numEigenVectors]                 
                    dist = np.sqrt(np.sum(p2pVec**2))    

                    simMat[keyFrameId, fid] = -dist
                    simMat[fid, keyFrameId] = simMat[keyFrameId, fid]

                print '%.3d' % keyFrameId,   

            print '\ntook {} seconds'.format(time.time() - tSimMat)

        else:
            raise ValueError('Invalid similarity method %s' % self.similarityMethod)
        
        # store results
        self.simMat_ = simMat
        
        if self.similarityMethod == 'PCA': 
            self.pca_ = pca
            self.X_proj_ = X_proj        

        # return results    
        return simMat
    
    def _estimate_phase(self, imAnalyze):

        # Step-1: compute inter-frame similarity matrix
        simMat = self._compute_frame_similarity(imAnalyze)
        
        # trend extraction using loess (local regression)
        def trendLowess(a, frac=0.3):
            return sm.nonparametric.lowess(a, np.arange(len(a)), 
                                           frac=frac, is_sorted=True)[:,1]

        # find the optimal key frame and use it to decompose 
        spectralEntropy = np.zeros((simMat.shape[0], 1)) 

        simMat_Trend = np.zeros_like(simMat)
        simMat_Seasonal = np.zeros_like(simMat)

        for fid in range(simMat.shape[0]):

            ts = simMat[fid,]

            # decompose into trend and seasonal parts
            ts_trend = trendLowess(ts)

            if self.detrend_mode == '-':
                ts_seasonal = ts - ts_trend
            elif self.detrend_mode == '/':
                ts_seasonal = ts / ts_trend

            if self.similarityMethod == 'PCA':
                ts_seasonal *= -1
    
            # compute periodogram entropy of the seasonal part
            freq, power = scipy.signal.periodogram(ts_seasonal)

            # store result
            simMat_Trend[fid,] = ts_trend
            simMat_Seasonal[fid,] = ts_seasonal
            spectralEntropy[fid] = scipy.stats.entropy(power)

        fid_best = np.argmin(spectralEntropy)    
        ts = simMat[fid_best, ]
        ts_trend = simMat_Trend[fid_best, ]
        ts_seasonal = simMat_Seasonal[fid_best, ]

        # estimate period from the periodogram
        freq, power = scipy.signal.periodogram(ts_seasonal)
        maxPowerLoc = np.argmax(power)
        period = 1.0/freq[maxPowerLoc]
        print "Estimated period = %.2f frames" % period
        
        #beatsPerMinute = period * 60.0 / framesPerSecDownsmp
        #print "beats per minute at %f fps = %f" % (framesPerSecDownsmp, beatsPerMinute)

        # compute analytic signal, instantaneous phase and amplitude
        ts_analytic = scipy.signal.hilbert(ts_seasonal - ts_seasonal.mean())
        ts_instaamp = np.abs(ts_analytic)
        ts_instaphase = np.arctan2(np.imag(ts_analytic), np.real(ts_analytic))        
        
        # store results
        self.simMat_Trend_ = simMat_Trend
        self.simMat_Seasonal_ = simMat_Seasonal
        self.spectralEntropy_ = spectralEntropy
        
        self.fid_best_ = fid_best
        self.ts_ = ts
        self.ts_trend_ = ts_trend
        self.ts_seasonal_ = ts_seasonal
        self.period_ = period
        
        self.ts_analytic_ = ts_analytic
        self.ts_instaamp_ = ts_instaamp
        self.ts_instaphase_ = ts_instaphase
        self.ts_instaphase_nmzd_ = (ts_instaphase + np.pi) / (2 * np.pi)        
        
    def process(self, imInput):

        tProcessing = time.time()
        
        # Step-1: Suppress noise using low-rank plus sparse decomposition
        print '\n>> Step-1: Suppressing noise using low-rank plus sparse decomposition ...\n'
        
        tDenoising = time.time()
        
        self._denoise(imInput)
        
        print '\nNoise suppression took a total of %.2f seconds' % (time.time() - tDenoising)
        
        # Step-2: Estimate intra-period phase
        print '\n>> Step-2: Estimating instantaneous phase ...\n'
        
        tPhaseEstimation = time.time()        
        
        self._estimate_phase(self.imLowRank_)
        
        print '\nPhase estimation took a total of %.2f seconds' % (time.time() - tPhaseEstimation)
        
        # Done processing
        print '\n>> Done processing ... took a total of %.2f seconds' % (time.time() - tProcessing)
        
    def generateSinglePeriodVideo(self, numOutFrames, sigmaGKRFactor=2, phaseRange=[0,1], imInput=None):
        
        # define gaussian
        def gauss_kernel(x, mu, sigma):  
            r = (normalizeAngles(x - mu, [-np.pi, np.pi]))
            return np.exp( -r**2 / (2.0 * sigma**2) )

        # validate phase argument
        if not (len(phaseRange) == 2 and 
                np.all(phaseRange) >= 0 and np.all(phaseRange) <= 1 and                
                phaseRange[0] < phaseRange[1]):
            raise ValueError('Invalid phase range')
            
        # compute sigmaGKR 
        sigmaGKR = sigmaGKRFactor * np.mean( np.abs(np.diff(np.sort(self.ts_instaphase_))) )
        print 'sigmaGKR = ', sigmaGKR

        # if imInput is not provided use imLowRank to generate the video
        if not imInput:
            imInput = self.imLowRank_
            
        # generate video    
        X = np.reshape(imInput, (np.prod(imInput.shape[:2]), imInput.shape[2])).T
        
        phaseRange = [np.pi * (2 * i - 1) for i in phaseRange] # map to [-pi, pi]        
        phaseVals = np.linspace(phaseRange[0], phaseRange[1], numOutFrames)

        imOnePeriodVideo = []
        prevPercent = 0

        for fid in range(numOutFrames):

            curPhase = phaseVals[fid]

            # generate frame by rbf interpolation
            w = gauss_kernel(self.ts_instaphase_, curPhase, sigmaGKR).T  

            imCurFrame = np.reshape(np.dot(w / w.sum(), X), imInput.shape[:2])

            # add to video
            if fid == 0:
                imOnePeriodVideo = imCurFrame
            else:
                imOnePeriodVideo = np.dstack((imOnePeriodVideo, imCurFrame))

            # update progress
            curPercent = np.floor(100.0*fid/numOutFrames)
            if curPercent > prevPercent:
                prevPercent = curPercent
                print '%.2d%%' % curPercent,         
      
        return imOnePeriodVideo
