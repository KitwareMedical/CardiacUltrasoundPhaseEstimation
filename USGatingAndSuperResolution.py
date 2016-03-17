import numpy as np
import scipy.ndimage
import time
import sys
import sklearn.decomposition
import statsmodels.api as sm
import angles
import cv2
import SimpleITK as sitk
import registration_utilities as ru
import registration_callbacks as rc

sys.path.insert(0, 'pyLAR')
import core.ialm


def normalizeAngles(angleList, angle_range):
    return np.array([angles.normalize(i, angle_range[0], angle_range[1])
                     for i in angleList])

def phaseDist(p1, p2, maxPhase=1.0):
    
    flagScalarInput = False
    
    if np.isscalar(p1) and np.isscalar(p2):
        flagScalarInput = True
        p1 = np.array(p1)
        p2 = np.array(p2)
    
    modDiff = np.array(np.abs(p2 - p1) % maxPhase)
    
    flagDiffGTMid = modDiff > 0.5 * maxPhase
    modDiff[flagDiffGTMid] = maxPhase - modDiff[flagDiffGTMid]
    
    if flagScalarInput:
        return np.asscalar(modDiff)
    else:
        return modDiff
    

def phaseDiff(phaseArr, maxPhase=1.0):
    n = len(phaseArr)
    return phaseDist(phaseArr[:n-1], phaseArr[1:])
    

def ncorr(imA, imB):
    
    imA = (imA - imA.mean()) / imA.std()
    imB = (imB - imB.mean()) / imB.std()
    
    return np.mean(imA * imB)    


def rmse(imA, imB):    
    return np.sqrt(np.mean((imA - imB)**2)) 
    
def compute_mean_consec_frame_rmse(imInput):
    
    mean_rmse = 0.0
    
    for i in range(imInput.shape[2]-1):
        imCurFrame = imInput[:, :, i]
        imNextFrame = imInput[:, :, i+1]
        rmse = np.sqrt(np.mean((imNextFrame.flatten() - imCurFrame.flatten())**2))
        mean_rmse += rmse
    
    mean_rmse /= (imInput.shape[2] - 1)
    
    return rmse

def compute_mean_consec_frame_ncorr(imInput):
    
    mean_ncorr = 0.0
    
    for i in range(imInput.shape[2]-1):
        imCurFrame = imInput[:, :, i]
        imNextFrame = imInput[:, :, i+1]
        mean_ncorr += ncorr(imCurFrame, imNextFrame)
    
    mean_ncorr /= (imInput.shape[2] - 1)
    
    return mean_ncorr

def config_framegen_using_linear_interpolation():

    return {'name': 'linear_interpolation',
            'params': {}
           }

def config_framegen_using_kernel_regression(sigmaGKRFactor = 2):

    return {'name': 'kernel_regression',
            'params': {'sigmaGKRFactor': sigmaGKRFactor}
           }

def config_framegen_using_optical_flow(pyr_scale=0.5, levels=4, 
                                       winsizeFactor=0.5, iterations=3, 
                                       poly_n=7, poly_sigma=1.5,
                                       flags=0):

    return {'name': 'optical_flow',
            'params': 
                {
                 'pyr_scale': pyr_scale,
                 'levels': levels,
                 'winsizeFactor': winsizeFactor,                
                 'iterations': iterations,
                 'poly_n': poly_n,
                 'poly_sigma': poly_sigma,
                 'flags': flags
                }
            }

def frame_gen_optical_flow(im1, im2, alpha,
                           pyr_scale=0.5, levels=4, 
                           winsizeFactor=0.5, iterations=3, 
                           poly_n=7, poly_sigma=1.5,
                           flags=0):
    
    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR, borderValue=np.median(img[:, 0]))
        return res  

    winsize = np.max(np.ceil(winsizeFactor * np.array(im1.shape[:2]))).astype('int')  
    
    flowPrev = cv2.calcOpticalFlowFarneback(im1, im2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    
    flowNext = cv2.calcOpticalFlowFarneback(im2, im1, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
 
    imWarp1 = warp_flow(im1, flow1 * alpha)
                                  
    imWarp2 = warp_flow(im2, flow2 * (1 - alpha))
    
    imResult = 0.5 * (imWarp1 + imWarp2)
    
    return imResult

def config_framegen_using_bspline_registration(gridSpacingFactor=0.15, gradConvTol=1e-4, 
                                               affineIter=50, bsplineIter=50):
    
    return {'name': 'bspline_registration',
            'params': 
                {
                 'gridSpacingFactor': gridSpacingFactor, 
                 'gradConvTol': gradConvTol, 
                 'affineIter': affineIter,
                 'bsplineIter': bsplineIter
                }
            }

def frame_gen_bspline_registration(im1, im2, alpha, 
                      gridSpacingFactor=0.15, gradConvTol=1e-4, 
                      affineIter=50, bsplineIter=50, debug=False):
    
    moving_image = sitk.GetImageFromArray(im1.astype('float'))
    fixed_image = sitk.GetImageFromArray(im2.astype('float'))
    
    #
    # affine registration
    #
    if debug:
        print '>>> Performing affine registration ...'
    
    affineReg = sitk.ImageRegistrationMethod()
    
    # metric
    affineReg.SetMetricAsMeanSquares()
    affineReg.SetMetricSamplingStrategy(affineReg.RANDOM)
    affineReg.SetMetricSamplingPercentage(0.01)
    
    # interpolator
    affineReg.SetInterpolator(sitk.sitkLinear)
    
    # transform
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, 
                                                          sitk.Similarity2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    affineReg.SetInitialTransform(initial_transform)
    
    # optimizer
    #affineReg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50, estimateLearningRate=affineReg.Once)   
    affineReg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=gradConvTol, maximumNumberOfIterations=affineIter)
    affineReg.SetOptimizerScalesFromPhysicalShift()
    
    # multi-resolution setup
    affineReg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    affineReg.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    affineReg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
   
    # connect all of the observers so that we can perform plotting during registration
    if debug:
        affineReg.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
        affineReg.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
        affineReg.AddCommand(sitk.sitkMultiResolutionIterationEvent, rc.metric_update_multires_iterations) 
        affineReg.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_plot_values(affineReg))
    
    # Execute
    affine_transform = affineReg.Execute(fixed_image, moving_image)
    
    if debug:
        print('Final metric value for affine registration: {0}'.format(affineReg.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(affineReg.GetOptimizerStopConditionDescription()))

    #
    # Bspline registration
    #
    if debug:
        print '>>> Performing bspline registration ...'
    
    bsplineReg = sitk.ImageRegistrationMethod()
    
    # metric
    bsplineReg.SetMetricAsMeanSquares()
    bsplineReg.SetMetricSamplingStrategy(affineReg.RANDOM)
    bsplineReg.SetMetricSamplingPercentage(0.01)
    
    # interpolator
    bsplineReg.SetInterpolator(sitk.sitkLinear)
    
    # initial transform
    bsplineReg.SetMovingInitialTransform(affine_transform)
    mesh_size = [int(gridSpacingFactor * sz) for sz in fixed_image.GetSize()]
    if debug:
        print mesh_size
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size, order=3)
      
    bsplineReg.SetInitialTransform(initial_transform)
    
    # optimizer
    #bsplineReg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=bsplineReg.Once)   
    bsplineReg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=gradConvTol, maximumNumberOfIterations=bsplineIter)
    bsplineReg.SetOptimizerScalesFromPhysicalShift()
    
    # multi-resolution setup
    bsplineReg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    bsplineReg.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    bsplineReg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
   
    # connect all of the observers so that we can perform plotting during registration
    if debug:
        bsplineReg.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
        bsplineReg.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
        bsplineReg.AddCommand(sitk.sitkMultiResolutionIterationEvent, rc.metric_update_multires_iterations) 
        bsplineReg.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_plot_values(bsplineReg))
    
    # Execute
    bspline_transform = bsplineReg.Execute(fixed_image, moving_image)
    
    if debug:
        print('Final metric value: {0}'.format(bsplineReg.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(bsplineReg.GetOptimizerStopConditionDescription()))
    
    # compose affine and bspline transform
    final_transform = sitk.Transform(bspline_transform)
    final_transform.AddTransform(affine_transform)
    
    # convert to displacement field image
    disp_field_converter = sitk.TransformToDisplacementFieldFilter()
    disp_field_converter.SetReferenceImage(fixed_image)
    
    disp_field_image = disp_field_converter.Execute(final_transform)
    
    # module displacement field image
    disp_field_image_fwd = sitk.GetImageFromArray(alpha * sitk.GetArrayFromImage(disp_field_image), isVector=True)
    disp_field_image_bck = sitk.GetImageFromArray((1 - alpha) * sitk.GetArrayFromImage(disp_field_image), isVector=True)
    
    # transform moving image
    defaultVal = np.double(np.median(im1[:, 0]))
    
    final_transform_fwd = sitk.DisplacementFieldTransform(disp_field_image_fwd)
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_fwd,
                                     sitk.sitkLinear, defaultVal, fixed_image.GetPixelIDValue())    

    # transform fixed image
    defaultVal = np.double(np.median(im2[:, 0]))
    final_transform_bck = sitk.DisplacementFieldTransform(disp_field_image_bck)
    fixed_resampled = sitk.Resample(fixed_image, fixed_image, final_transform_bck,
                                    sitk.sitkLinear, defaultVal, fixed_image.GetPixelIDValue())    
    
    imResult = 0.5 * (sitk.GetArrayFromImage(fixed_resampled) + sitk.GetArrayFromImage(moving_resampled))
    
    return imResult, final_transform    

class USGatingAndSuperResolution(object):

    def __init__(self, 
                 # noise suppression parameters
                 medianFilterSize=3, xyDownsamplingFactor=1, gammaFactor=1, lrconvtol=1e-05,
                 # phase estimation parameters
                 similarityMethod='PCA', pca_n_components=0.99,
                 lowessFrac=0.3, detrend_mode='/'
                 ):
        
        # noise suppression parameters
        self.medianFilterSize = medianFilterSize
        self.gammaFactor = gammaFactor
        self.lrconvtol = lrconvtol
        
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

        # reduce xy size to speed up low-rank + sparse decomposition
        if self.xyDownsamplingFactor < 1:
            imInputForLowrank = scipy.ndimage.interpolation.zoom(imInputForLowrank, 
                                                                 (self.xyDownsamplingFactor, 
                                                                  self.xyDownsamplingFactor, 1))  

        # create a matrix D where each column represents data at one time point
        D = np.reshape(imInputForLowrank, (np.prod(imInputForLowrank.shape[:2]), imInputForLowrank.shape[2]))    

        # perform low-rank plus sparse decomposition on D
        tRPCA = time.time()

        gamma = self.gammaFactor / np.sqrt(np.max(D.shape))

        res = core.ialm.recover(D, gamma, tol=self.lrconvtol)

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
            raise ValueError('Invalid similarity method %s' 
                             % self.similarityMethod)
        
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
                                           frac=frac, is_sorted=True)[:, 1]

        # find the optimal key frame and use it to decompose
        spectralEntropy = np.zeros((simMat.shape[0], 1))

        simMat_Trend = np.zeros_like(simMat)
        simMat_Seasonal = np.zeros_like(simMat)

        for fid in range(simMat.shape[0]):

            ts = simMat[fid, ]

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
            simMat_Trend[fid, ] = ts_trend
            simMat_Seasonal[fid, ] = ts_seasonal
            spectralEntropy[fid] = scipy.stats.entropy(power)

        fid_best = np.argmin(spectralEntropy)    
        ts = simMat[fid_best, ]
        ts_trend = simMat_Trend[fid_best, ]
        ts_seasonal = simMat_Seasonal[fid_best, ]
        print "Chose frame %d as key frame" % fid_best

        # estimate period from the periodogram
        freq, power = scipy.signal.periodogram(ts_seasonal)
        maxPowerLoc = np.argmax(power)
        period = 1.0/freq[maxPowerLoc]
        print "Estimated period = %.2f frames" % period
        print "Estimated number of periods = %.2f" % (ts_seasonal.size / period) 
        
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
    
    
    def setInput(self, imInput):
        
        self.imInput_ = imInput
        
        
    def process(self):

        tProcessing = time.time()        
        
        print 'Input video size: ', self.imInput_.shape
        
        # Step-1: Suppress noise using low-rank plus sparse decomposition
        print '\n>> Step-1: Suppressing noise using low-rank plus sparse decomposition ...\n'
        
        tDenoising = time.time()
        
        self._denoise(self.imInput_)
        
        print '\nNoise suppression took a total of %.2f seconds' % (time.time() - tDenoising)
        
        # Step-2: Estimate intra-period phase
        print '\n>> Step-2: Estimating instantaneous phase ...\n'
        
        tPhaseEstimation = time.time()        
        
        self._estimate_phase(self.imLowRank_)
        
        print '\nPhase estimation took a total of %.2f seconds' % (time.time() - tPhaseEstimation)
        
        # Done processing
        print '\n>> Done processing ... took a total of %.2f seconds' % (time.time() - tProcessing)
        
    def generateFramesFromPhaseValues(self, phaseVals,
                                      imInput=None, method=None):                                      
 
        # validate phase vals
        phaseVals = np.array(phaseVals)
        
        if np.any(phaseVals < 0) or np.any(phaseVals > 1):
            raise ValueError('Invalid phase values')

        # set imInput
        if imInput is None:
            imInput = self.imInput_

        # generate frames
        imOutputVideo = []
        
        numOutFrames = len(phaseVals)
        
        if method is None:
            method = config_framegen_using_bspline_registration()
            
        if method['name'] == 'kernel_regression':   
            
            # compute sigmaGKR 
            sigmaGKRFactor = method['params']['sigmaGKRFactor']
            #sigmaGKR = sigmaGKRFactor * np.mean(np.abs(np.diff(np.sort(self.ts_instaphase_))))
            #sigmaGKR = sigmaGKRFactor * np.mean(np.abs(np.diff(np.sort(self.ts_instaphase_nmzd_))))
            sigmaGKR = sigmaGKRFactor * np.mean(phaseDiff(np.sort(self.ts_instaphase_nmzd_)))
            print 'sigmaGKR = ', sigmaGKR
            
            # define gaussian
            def gauss_kernel(x, mu, sigma):
                #r = (normalizeAngles(x - mu, [-np.pi, np.pi]))
                #r = (x - mu) % 1
                r = phaseDist(mu, x)
                return np.exp(-r**2 / (2.0 * sigma**2))            
            
            X = np.reshape(imInput, (np.prod(imInput.shape[:2]), imInput.shape[2])).T
        
        prevPercent = 0
        
        for fid in range(numOutFrames):

            curPhase = phaseVals[fid]
            curPhaseAngle = np.pi * (2 * curPhase - 1)
            
            if method['name'] == 'kernel_regression':
                
                # generate frame by rbf interpolation
                #w = gauss_kernel(self.ts_instaphase_, curPhaseAngle, sigmaGKR).T
                w = gauss_kernel(self.ts_instaphase_nmzd_, curPhase, sigmaGKR).T
                imCurFrame = np.reshape(np.dot(w / w.sum(), X), imInput.shape[:2])
                
            elif method['name'] in ['optical_flow', 'bspline_registration', 'linear_interpolation']:    
            
                # generate frame using optical flow
                prevPhaseInd = np.argmin((curPhase - self.ts_instaphase_nmzd_) % 1)
                prevPhase = self.ts_instaphase_nmzd_[prevPhaseInd]
                
                nextPhaseInd = np.argmin((self.ts_instaphase_nmzd_ - curPhase) % 1)
                nextPhase = self.ts_instaphase_nmzd_[nextPhaseInd]                
                
                prevPhaseDist = phaseDist(prevPhase, curPhase)
                nextPhaseDist = phaseDist(curPhase, nextPhase)
                totalPhaseDist = prevPhaseDist + nextPhaseDist
                
                alpha = prevPhaseDist / totalPhaseDist
                
                imPrevFrame = imInput[:, :, prevPhaseInd]
                imNextFrame = imInput[:, :, nextPhaseInd]
                
                if method['name'] == 'optical_flow':
                    
                    imCurFrame = frame_gen_optical_flow(imPrevFrame, imNextFrame, alpha, **(method['params']))
                    
                elif method['name'] == 'bspline_registration':
                    
                    imCurFrame, _ = frame_gen_bspline_registration(imPrevFrame, imNextFrame, alpha, **(method['params']))
                    
                elif method['name'] == 'linear_interpolation':
                    
                    imCurFrame = (1-alpha) * imPrevFrame + alpha * imNextFrame

            else:
                raise ValueError('Invalid method - %s' % method['name'])

            # add to video
            if fid == 0:
                imOutputVideo = imCurFrame
            else:
                imOutputVideo = np.dstack((imOutputVideo, imCurFrame))

            # update progress
            curPercent = np.floor(100.0*fid/numOutFrames)
            if curPercent > prevPercent:
                prevPercent = curPercent
                print '%.2d%%' % curPercent,
        
        print '\n'
        return imOutputVideo

    
    def generateVideo(self, numOutFrames, phaseRange,
                      imInput=None, method=None):                                                    
                      
        # validate phase argument
        if not (len(phaseRange) == 2 and
                np.all(phaseRange >= 0) and np.all(phaseRange <= 1) and
                phaseRange[0] < phaseRange[1]):
            raise ValueError('Invalid phase range')
        
        # generate video
        phaseVals = np.linspace(phaseRange[0], phaseRange[1], numOutFrames)
        
        return self.generateFramesFromPhaseValues(phaseVals,
                                                  imInput=imInput, method=method)

    def generateSinglePeriodVideo(self, numOutFrames, 
                                  imInput=None, method=None):
        
        phaseRange = np.array([0, 1])
        return self.generateVideo(numOutFrames, phaseRange,
                                  imInput=imInput, method=method)