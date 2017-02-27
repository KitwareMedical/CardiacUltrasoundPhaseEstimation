import numpy as np
import scipy.ndimage
import scipy.interpolate
import time
import sys
import sklearn.decomposition
import statsmodels.api as sm
import angles
import cv2
import SimpleITK as sitk
import registration_utilities as ru
import registration_callbacks as rc
import skimage.measure
import medpy.metric.image
import functools

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.nonparametric.smoothers_lowess import lowess

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
        cur_rmse = np.sqrt(np.mean((imNextFrame.flatten() - imCurFrame.flatten())**2))
        mean_rmse += cur_rmse

    mean_rmse /= (imInput.shape[2] - 1.0)

    return mean_rmse


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


def config_framegen_using_kernel_regression(sigmaPhaseFactor = 0.5,
                                            sigmaSimilarityFactor=None,
                                            stochastic=False):

    suffix = '_phase'

    if sigmaSimilarityFactor is not None:
        suffix += '_sim'

    if stochastic:
        suffix += '_stochastic'

    return {'name': 'kernel_regression' + suffix,
            'params':
                {
                    'sigmaPhaseFactor': sigmaPhaseFactor,
                    'sigmaSimilarityFactor': sigmaSimilarityFactor,
                    'stochastic': stochastic
                }
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
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR,
                        borderValue=np.median(img[:, 0]))
        return res  

    winsize = np.max(np.ceil(winsizeFactor * np.array(im1.shape[:2]))).astype('int')  

    flowFwd = cv2.calcOpticalFlowFarneback(im1, im2, pyr_scale, levels,
                                           winsize, iterations, poly_n,
                                           poly_sigma, flags)

    flowBwd = cv2.calcOpticalFlowFarneback(im2, im1, pyr_scale, levels,
                                           winsize, iterations, poly_n,
                                           poly_sigma, flags)

    imWarpFwd = warp_flow(im1, flowFwd * alpha)

    imWarpBwd = warp_flow(im2, flowBwd * (1 - alpha))

    imResult = 0.5 * (imWarpFwd + imWarpBwd)

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


def register_rigid(im_fixed, im_moving, iter=50, debug=False):

    moving_image = sitk.GetImageFromArray(im_moving.astype('float'))
    fixed_image = sitk.GetImageFromArray(im_fixed.astype('float'))

    reg = sitk.ImageRegistrationMethod()

    # metric
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)

    # interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg.SetInitialTransform(initial_transform)

    # optimizer
    #reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50, estimateLearningRate=affineReg.Once)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4,
                             maximumNumberOfIterations=500)
    reg.SetOptimizerScalesFromPhysicalShift()

    # multi-resolution setup
    reg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # connect all of the observers so that we can perform plotting during registration
    if debug:
        reg.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
        reg.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
        reg.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                       rc.metric_update_multires_iterations)
        reg.AddCommand(sitk.sitkIterationEvent,
                       lambda: rc.metric_plot_values(reg))

    # Execute
    tfm = reg.Execute(fixed_image, moving_image)

    # post reg analysis
    if debug:
        print('Final metric value for affine registration: {0}'.format(
            reg.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(
            reg.GetOptimizerStopConditionDescription()))

    # transform moving image
    moving_resampled = sitk.Resample(moving_image, fixed_image, tfm,
                                     sitk.sitkLinear, np.double(im_fixed.min()),
                                     fixed_image.GetPixelIDValue())

    return sitk.GetArrayFromImage(moving_resampled)


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
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    affineReg.SetInitialTransform(initial_transform)

    # optimizer
    #affineReg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50, estimateLearningRate=affineReg.Once)   
    affineReg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=gradConvTol,
                                   maximumNumberOfIterations=affineIter)
    affineReg.SetOptimizerScalesFromPhysicalShift()

    # multi-resolution setup
    affineReg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    affineReg.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    affineReg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # connect all of the observers so that we can perform plotting during registration
    if debug:
        affineReg.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
        affineReg.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
        affineReg.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                             rc.metric_update_multires_iterations)
        affineReg.AddCommand(sitk.sitkIterationEvent,
                             lambda: rc.metric_plot_values(affineReg))

    # Execute
    affine_transform = affineReg.Execute(fixed_image, moving_image)

    if debug:
        print('Final metric value for affine registration: {0}'.format(
            affineReg.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(
            affineReg.GetOptimizerStopConditionDescription()))

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
    bsplineReg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=gradConvTol,
                                    maximumNumberOfIterations=bsplineIter)
    bsplineReg.SetOptimizerScalesFromPhysicalShift()

    # multi-resolution setup
    bsplineReg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    bsplineReg.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    bsplineReg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # connect all of the observers so that we can perform plotting during registration
    if debug:
        bsplineReg.AddCommand(sitk.sitkStartEvent, rc.metric_start_plot)
        bsplineReg.AddCommand(sitk.sitkEndEvent, rc.metric_end_plot)
        bsplineReg.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                              rc.metric_update_multires_iterations)
        bsplineReg.AddCommand(sitk.sitkIterationEvent,
                              lambda: rc.metric_plot_values(bsplineReg))

    # Execute
    bspline_transform = bsplineReg.Execute(fixed_image, moving_image)

    if debug:
        print('Final metric value: {0}'.format(bsplineReg.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(
            bsplineReg.GetOptimizerStopConditionDescription()))

    # compose affine and bspline transform
    final_transform = sitk.Transform(bspline_transform)
    final_transform.AddTransform(affine_transform)

    # convert to displacement field image
    disp_field_converter = sitk.TransformToDisplacementFieldFilter()
    disp_field_converter.SetReferenceImage(fixed_image)

    disp_field_image = disp_field_converter.Execute(final_transform)

    # module displacement field image
    disp_field_image_fwd = sitk.GetImageFromArray(
        alpha * sitk.GetArrayFromImage(disp_field_image), isVector=True)
    disp_field_image_bck = sitk.GetImageFromArray(
        (1 - alpha) * sitk.GetArrayFromImage(disp_field_image), isVector=True)

    # transform moving image
    defaultVal = np.double(np.median(im1[:, 0]))

    final_transform_fwd = sitk.DisplacementFieldTransform(disp_field_image_fwd)
    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     final_transform_fwd,
                                     sitk.sitkLinear, defaultVal,
                                     fixed_image.GetPixelIDValue())

    # transform fixed image
    defaultVal = np.double(np.median(im2[:, 0]))
    final_transform_bck = sitk.DisplacementFieldTransform(disp_field_image_bck)
    fixed_resampled = sitk.Resample(fixed_image, fixed_image,
                                    final_transform_bck,
                                    sitk.sitkLinear, defaultVal,
                                    fixed_image.GetPixelIDValue())

    imResult = 0.5 * (sitk.GetArrayFromImage(fixed_resampled) +
                      sitk.GetArrayFromImage(moving_resampled))

    return imResult, final_transform


def detrend_lowess(ts, frac=0.3, mode='-'):

    ts_trend = lowess(ts, np.arange(len(ts)),
                      frac=frac, is_sorted=True)[:, 1]

    if mode == '-':
        ts_seasonal = ts - ts_trend
    else:
        ts_seasonal = ts / ts_trend

    return ts_seasonal, ts_trend


def compute_instantaneous_phase(ts):

    ts_analytic = scipy.signal.hilbert(ts - ts.mean())
    ts_instaamp = np.abs(ts_analytic)
    ts_instaphase = np.arctan2(np.imag(ts_analytic), np.real(ts_analytic))
    ts_instaphase_nmzd = (ts_instaphase + np.pi) / (2 * np.pi)

    return ts_instaphase_nmzd, ts_instaamp

# define gaussian phase kernel
def gauss_phase_kernel(x, mu, sigma):
    #r = (normalizeAngles(x - mu, [-np.pi, np.pi]))
    #r = (x - mu) % 1
    r = phaseDist(mu, x)
    return np.exp(-r**2 / (2.0 * sigma**2))

# define gaussian similarity kernel
def gauss_similarity_kernel(x, mu, sigma):
    r = (x - mu)
    return np.exp(-r**2 / (2.0 * sigma**2))


class LowessRegression(object):

    def __init__(self, frac=0.2, it=3):

        self.frac_ = frac
        self.it_ = 3
    
    def fit(self, x, y, is_sorted=False):

        res_lowess = sm.nonparametric.lowess(y, x, 
                                             frac=self.frac_, it=self.it_,
                                             is_sorted=is_sorted)

        self.lowess_x_ = res_lowess[:, 0]
        self.lowess_y_ = res_lowess[:, 1]
        self.residual_mad_ = sm.robust.scale.mad(y - self.lowess_y_, center=0)

        self.lowess_interp_ = scipy.interpolate.interp1d(
            self.lowess_x_, self.lowess_y_,
            bounds_error=False, fill_value='extrapolate')

    def predict(self, x):

        return self.lowess_interp_(x)

    def residual_mad(self):

        return self.residual_mad_


class USGatingAndSuperResolution(object):

    def __init__(self,
                 # noise suppression parameters
                 median_filter_size=3, apply_lr_denoising = True,
                 lr_xy_downsampling=1, lr_gamma_factor=1, lr_conv_tol=1e-05,
                 # phase estimation parameters
                 similarity_method='pca', pca_n_components=0.99,
                 detrend_method='hp', lowess_frac=0.3, lowess_mode='-',
                 hp_lamda = 6400,
                 respiration_present=False, resp_phase_cutoff=0.2
                 ):

        # noise suppression parameters
        self.median_filter_size = median_filter_size

        self.apply_lr_denoising = apply_lr_denoising
        self.lr_gamma_factor = lr_gamma_factor
        self.lr_conv_tol = lr_conv_tol

        if not (lr_xy_downsampling > 0 and lr_xy_downsampling <= 1.0):
            raise ValueError('lr_xy_downsampling should in (0, 1]')
        self.lr_xy_downsampling = lr_xy_downsampling

        # phase estimation parameters
        if similarity_method not in ['ncorr', 'pca']:
            raise ValueError("Invalid similarity method. Must be ncorr or pca")
        self.similarity_method = similarity_method

        self.pca_n_components = pca_n_components

        if detrend_method not in ['lowess', 'hp']:
            raise ValueError("Invalid detrend method. Must be lowess or hp")
        self.detrend_method = detrend_method

        if lowess_mode not in ['/', '-']:
            raise ValueError("Invalid detrend mode. Must be '/' or '-'")            
        self.lowess_mode = lowess_mode

        if not (lowess_frac > 0 and lowess_frac <= 1.0):
            raise ValueError('lowess_frac should in (0, 1]')
        self.lowess_frac = lowess_frac

        self.hp_lamda = hp_lamda

        self.respiration_present = respiration_present
        self.resp_phase_cutoff = resp_phase_cutoff

    def _denoise(self, imInput):

        imInputDenoised = imInput

        # denoise using median filter if requested
        if self.median_filter_size > 0:

            imInputDenoised = scipy.ndimage.filters.median_filter(
                imInputDenoised,
                (self.median_filter_size, self.median_filter_size, 1)
            )

        if self.apply_lr_denoising:

            # reduce xy size to speed up low-rank + sparse decomposition
            if self.lr_xy_downsampling < 1:
                imInputDenoised = scipy.ndimage.interpolation.zoom(
                    imInputDenoised,
                    (self.lr_xy_downsampling, self.lr_xy_downsampling, 1)
                )

            # create a matrix D where each column represents one video frame
            D = np.reshape(imInputDenoised, (np.prod(imInputDenoised.shape[:2]),
                                             imInputDenoised.shape[2]))

            # perform low-rank plus sparse decomposition on D
            tRPCA = time.time()

            gamma = self.lr_gamma_factor / np.sqrt(np.max(D.shape))

            res = core.ialm.recover(D, gamma, tol=self.lr_conv_tol)

            D_lowRank = np.array( res[0] )
            D_sparse = np.array( res[1] )

            imD = np.reshape(D, imInputDenoised.shape)
            imLowRank = np.reshape(D_lowRank, imInputDenoised.shape)
            imSparse = np.reshape(D_sparse, imInputDenoised.shape)

            if self.lr_xy_downsampling < 1:

                # restore result to original size
                zoomFactor = np.array(imInput.shape).astype('float') / \
                             np.array(imInputDenoised.shape)
                imD = scipy.ndimage.interpolation.zoom(imD, zoomFactor)
                imLowRank = scipy.ndimage.interpolation.zoom(imLowRank,
                                                             zoomFactor)
                imSparse = scipy.ndimage.interpolation.zoom(imSparse,
                                                            zoomFactor)

            print 'Low-rank plus sparse decomposition took {} seconds'.format(
                time.time() - tRPCA)

            imInputDenoised = imLowRank

            # store results
            self.imD_ = imD
            self.imLowRank_ = imLowRank
            self.imSparse_ = imSparse

        self.imInputDenoised_ = imInputDenoised

    def _compute_frame_similarity(self, imAnalyze):

        # Compute similarity of each time point with all other time points
        simMat = np.zeros((imAnalyze.shape[-1], imAnalyze.shape[-1]))

        if self.similarity_method == 'ncorr':

            print '\nComputing similarity using normalized correlation ... ',

            tSimMat = time.time()

            X = np.reshape(imAnalyze, (np.prod(imAnalyze.shape[:2]), imAnalyze.shape[-1]))
            X = (X - X.mean(0)) / X.std(0)
            simMat = np.dot(X.T, X) / X.shape[0]    

            print 'took {} seconds'.format(time.time() - tSimMat)

        elif self.similarity_method == 'pca':

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
            print '%d eigen vectors used to cover %.2f%% of variance' % (
                numEigenVectors, self.pca_n_components * 100)

            # Compute similarity of key frame with all the other frames
            print '\nComputing similarity as -ve distance in pca space ... '

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
                             % self.similarity_method)

        # store results
        self.simMat_ = simMat

        if self.similarity_method == 'pca':
            self.pca_ = pca
            self.X_proj_ = X_proj        

        # return results    
        return simMat

    def _estimate_phase(self, imAnalyze):

        # Step-1: compute inter-frame similarity matrix
        simMat = self._compute_frame_similarity(imAnalyze)

        # find the optimal key frame and use it to decompose
        spectralEntropy = np.zeros((simMat.shape[0], 1))

        simMat_Trend = np.zeros_like(simMat)
        simMat_Seasonal = np.zeros_like(simMat)

        for fid in range(simMat.shape[0]):

            ts = simMat[fid, ]

            # decompose into trend and seasonal parts
            if self.detrend_method == 'lowess':

                # lowess regression
                ts_seasonal, ts_trend = detrend_lowess(ts,
                                                       frac=self.lowess_frac,
                                                       mode=self.lowess_mode)

            else:

                # hoedrick-prescott filter
                ts_seasonal, ts_trend = hpfilter(ts, lamb=self.hp_lamda)

            # compute periodogram entropy of the seasonal part
            freq, power = scipy.signal.periodogram(ts_seasonal)

            # store result
            simMat_Trend[fid, ] = ts_trend
            simMat_Seasonal[fid, ] = ts_seasonal
            spectralEntropy[fid] = scipy.stats.entropy(power)

        fid_best = np.argmin(spectralEntropy)    
        ts = simMat[fid_best, :]
        ts_trend = simMat_Trend[fid_best, :]
        ts_seasonal = simMat_Seasonal[fid_best, :]

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
        ts_instaphase_nmzd = (ts_instaphase + np.pi) / (2 * np.pi)

        # estimate instantaneous phase of trend component - breathing
        ts_trend_analytic = scipy.signal.hilbert(ts_trend - ts_trend.mean())
        ts_trend_instaamp = np.abs(ts_trend_analytic)
        ts_trend_instaphase = np.arctan2(np.imag(ts_trend_analytic), np.real(ts_trend_analytic)) 
        ts_trend_instaphase_nmzd = (ts_trend_instaphase + np.pi) / (2 * np.pi)    

        # learn mapping from phase to similarity
        resp_ind = []

        if self.respiration_present:  # is set to True when breathing is present

            # identify frames with bad influence by respiration
            w = self.resp_phase_cutoff
            resp_ind = np.argwhere(
                np.logical_or(ts_trend_instaphase_nmzd < w,
                              ts_trend_instaphase_nmzd > 1.0 - w)
            ).ravel()

            print 'Frames with bad respiration influence = %.2f%%' % (
                100.0 * len(resp_ind) / len(ts))

        # find similarity bounds at each phase using lowess
        phaseord_est = np.argsort(ts_instaphase_nmzd)

        phaseord_est_wout_resp = [fid for fid in phaseord_est
                                  if fid not in resp_ind]

        fid_lowess = fid_best
        # fid_lowess = phaseord_est_wout_resp[0]

        assert(fid_lowess not in resp_ind)

        ph_wout_resp = ts_instaphase_nmzd[phaseord_est_wout_resp]
        sim_wout_resp = simMat[fid_lowess, phaseord_est_wout_resp]

        sim_lowess_reg = LowessRegression()
        sim_lowess_reg.fit(ph_wout_resp, sim_wout_resp, is_sorted=True)

        ts_lowess = sim_lowess_reg.predict(ts_instaphase_nmzd)

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
        self.ts_instaphase_nmzd_ = ts_instaphase_nmzd

        self.ts_trend_analytic_ = ts_trend_analytic
        self.ts_trenda_instaamp_ = ts_trend_instaamp
        self.ts_trend_instaphase_ = ts_trend_instaphase
        self.ts_trend_instaphase_nmzd_ = ts_trend_instaphase_nmzd

        self.resp_ind_ = resp_ind
        self.sim_lowess_reg_ = sim_lowess_reg
        self.ts_lowess_ = ts_lowess
        self.fid_lowess_ = fid_lowess

    def get_instaphase(self):

        return self.ts_instaphase_nmzd_

    def get_instaphase_trend(self):

        return self.ts_trend_instaphase_nmzd_

    def setInput(self, imInput):

        self.imInput_ = imInput

    def process(self):

        tProcessing = time.time()        

        print 'Input video size: ', self.imInput_.shape

        # Step-1: Suppress noise using low-rank plus sparse decomposition
        print '\n>> Step-1: Suppressing noise ...\n'

        tDenoising = time.time()

        self._denoise(self.imInput_)

        print '\nNoise suppression took %.2f seconds' % (time.time() -
                                                         tDenoising)

        # Step-2: Estimate intra-period phase
        print '\n>> Step-2: Estimating instantaneous phase ...\n'

        tPhaseEstimation = time.time()        

        self._estimate_phase(self.imInputDenoised_)

        print '\nPhase estimation took %.2f seconds' % (time.time() -
                                                        tPhaseEstimation)

        # Done processing
        print '\n>> Done processing ... took a total of %.2f seconds' % (
            time.time() - tProcessing)

    def validate_frame_generation(self, k=1, rounds=10, method=None,
                                  metric='ncorr', seed=1,
                                  mi_bins=16, k_mad=None, 
                                  exclude_similar_phase_frames=False):

        if k_mad is not None:

            valid_ind = [fid for fid in range(self.imInput_.shape[2])
                          if (np.abs(self.ts_[fid] - self.ts_lowess_[fid]) <
                              k_mad * self.sim_lowess_reg_.residual_mad())]

        else:

            valid_ind = [fid for fid in range(self.imInput_.shape[2])
                         if fid not in self.resp_ind_]


        mval = np.zeros(rounds)

        np.random.seed(seed)

        period = np.int(self.period_ + 0.5)
        
        for r in range(rounds):

            print r+1,

            # choose k frames randomly
            ksel_ind = np.random.choice(valid_ind, k, replace=False)
            ph_ksel = self.ts_instaphase_nmzd_[ksel_ind]
            
            # print '\t', zip(ksel_ind, ph_ksel)
            
            # Find similar phase frames in each cycle to exclude if requested
            sim_phase_ind = []
            
            if exclude_similar_phase_frames:

                for fid in ksel_ind:
                    
                    prev_ind = np.arange(fid-period, 0, -period, dtype='int')
                    next_ind = np.arange(fid+period, self.imInput_.shape[2], period, dtype='int')
                    
                    sim_phase_ind.extend(prev_ind)
                    sim_phase_ind.extend(next_ind)

                sim_phase_ind = np.unique(sim_phase_ind)

                # print '\t', zip(sim_phase_ind, self.ts_instaphase_nmzd_[sim_phase_ind])

            imExclude = self.imInput_[:, :, ksel_ind].astype('float')

            exclude_find = functools.reduce(np.union1d, (ksel_ind, self.resp_ind_, sim_phase_ind))
                                           
            imSynth = self.generateFramesFromPhaseValues(
                ph_ksel, method=method, show_progress=False,
                exclude_frames=exclude_find)

            cur_mval = 0.0

            for i in range(len(ksel_ind)):

                if metric == 'ncorr':

                    cur_mval += ncorr(imExclude[:, :, i], imSynth[:, :, i])

                elif metric == 'rmse':

                    cur_mval += rmse(imExclude[:, :, i], imSynth[:, :, i])

                elif metric == 'mad':

                    cur_mval += np.median(
                        np.abs(imExclude.ravel() - imSynth.ravel()))

                elif metric == 'mutual_information':

                    cur_mval += medpy.metric.image.mutual_information(
                        imExclude, imSynth, bins=mi_bins
                    )

                else:

                    raise ValueError('Invalid metric')

            cur_mval /= len(ksel_ind)

            mval[r] = cur_mval

        print '\n', mval

        return mval


    def generateFramesFromPhaseValues(self, phaseVals,
                                      imInput=None, method=None,
                                      exclude_frames=None,
                                      show_progress=True):
 
        # validate phase vals
        phaseVals = np.array(phaseVals)

        if np.any(phaseVals < 0) or np.any(phaseVals > 1):
            raise ValueError('Invalid phase values')

        phaseRecorded = self.ts_instaphase_nmzd_
        simRecorded = self.ts_

        # set imInput
        if imInput is None:
            imInput = self.imInput_

        # exclude the frames requested
        if exclude_frames is not None:

            fmask = np.ones(imInput.shape[2], dtype=bool)
            fmask[exclude_frames] = False

            imInput = imInput[:, :, fmask]
            phaseRecorded = phaseRecorded[fmask]
            simRecorded = simRecorded[fmask]

        # generate frames
        numOutFrames = len(phaseVals)

        imOutputVideo = np.zeros(imInput.shape[:2] + (numOutFrames, ))

        if method is None:
            method = config_framegen_using_kernel_regression()

        if method['name'].startswith('kernel_regression'):

            # compute sigmaPhase
            kPhase = method['params']['sigmaPhaseFactor']

            pdiff = phaseDiff(phaseRecorded)
            pstd = sm.robust.scale.mad(pdiff, center=0)

            sigmaPhase = kPhase * pstd

            # print 'sigmaPhase = ', sigmaPhase

            # compute sigmaSimilarity
            kSim = method['params']['sigmaSimilarityFactor']

            if kSim is not None:

                if exclude_frames is None:

                    sim_lowess_reg = self.sim_lowess_reg_

                else:

                    sim_lowess_reg = LowessRegression()
                    sim_lowess_reg.fit(phaseRecorded, simRecorded)

                sigmaSim = kSim * sim_lowess_reg.residual_mad()

                phaseSims = sim_lowess_reg.predict(phaseVals)

            X = np.reshape(imInput,
                           (np.prod(imInput.shape[:2]), imInput.shape[2])).T

        prevPercent = 0

        for fid in range(numOutFrames):

            curPhase = phaseVals[fid]

            if method['name'].startswith('kernel_regression'):

                # generate frame by rbf interpolation
                wPhase = gauss_phase_kernel(
                    phaseRecorded, curPhase, sigmaPhase).T

                w = wPhase

                if kSim is not None:

                    wSim = gauss_similarity_kernel(
                        simRecorded, phaseSims[fid], sigmaSim).T

                    w = wPhase * wSim

                w /= w.sum()

                if method['params']['stochastic']:

                    numRecordedFrames = X.shape[0]
                    numPixels = X.shape[1]

                    '''
                    imVote = np.zeros((256, numPixels))

                    for p in range(numPixels):
                        for f in range(numRecordedFrames):

                            imVote[X[f, p], p] += w[f]

                    d = np.zeros(numPixels)

                    for p in range(numPixels):

                        v = imVote[:, p]
                        v /= v.sum()

                        d[p] = np.random.choice(np.arange(256), size=1, p=v)

                    imCurFrame = np.reshape(d, imInput.shape[:2])
                    '''

                    # '''
                    fsel = np.random.choice(np.arange(numRecordedFrames),
                                            size=numPixels, p=w)

                    d = np.zeros(X.shape[1])

                    for i in range(d.size):
                        d[i] = X[fsel[i], i]

                    imCurFrame = np.reshape(d, imInput.shape[:2])
                    # '''

                    imCurFrame = scipy.ndimage.filters.median_filter(
                        imCurFrame, (3, 3))

                else:

                    imCurFrame = np.reshape(np.dot(w, X), imInput.shape[:2])


            elif method['name'] in ['optical_flow', 'bspline_registration',
                                    'linear_interpolation']:

                # find closest prev and next frame
                prevPhaseInd = np.argmin(
                    (curPhase - phaseRecorded) % 1)
                prevPhase = phaseRecorded[prevPhaseInd]

                nextPhaseInd = np.argmin(
                    (phaseRecorded - curPhase) % 1)
                nextPhase = phaseRecorded[nextPhaseInd]

                prevPhaseDist = phaseDist(prevPhase, curPhase)
                nextPhaseDist = phaseDist(curPhase, nextPhase)
                totalPhaseDist = prevPhaseDist + nextPhaseDist

                alpha = prevPhaseDist / totalPhaseDist

                imPrevFrame = imInput[:, :, prevPhaseInd]
                imNextFrame = imInput[:, :, nextPhaseInd]

                if method['name'] == 'optical_flow':

                    imCurFrame = frame_gen_optical_flow(
                        imPrevFrame, imNextFrame,
                        alpha, **(method['params']))

                elif method['name'] == 'bspline_registration':

                    imCurFrame, _ = frame_gen_bspline_registration(
                        imPrevFrame, imNextFrame, alpha, **(method['params']))

                elif method['name'] == 'linear_interpolation':

                    imCurFrame = (1-alpha) * imPrevFrame + alpha * imNextFrame

            else:
                raise ValueError('Invalid method - %s' % method['name'])

            # add to video
            imOutputVideo[:, :, fid] = imCurFrame

            # update progress
            if show_progress:

                curPercent = np.floor(100.0*fid/numOutFrames)
                if curPercent > prevPercent:
                    prevPercent = curPercent
                    print '%.2d%%' % curPercent,

        if show_progress:
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