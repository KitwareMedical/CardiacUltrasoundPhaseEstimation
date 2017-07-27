import os, time

import numpy as np

import scipy.signal
import scipy.misc
import scipy.ndimage.filters

import matplotlib.pyplot as plt

import PIL
from PIL import ImageDraw

import angles

import cv2
import SimpleITK as sitk


def cvShowImage(imDisp, strName, strAnnotation='', textColor=(0, 0, 255),
                resizeAmount=None):

    if resizeAmount is not None:
        imDisp = cv2.resize(imDisp.copy(), None, fx=resizeAmount,
                            fy=resizeAmount)

    imDisp = cv2.cvtColor(imDisp, cv2.COLOR_GRAY2RGB)

    if len(strAnnotation) > 0:
        cv2.putText(imDisp, strAnnotation, (10, 20), cv2.FONT_HERSHEY_PLAIN,
                    2.0, textColor, thickness=2)

    cv2.imshow(strName, imDisp)


def cvShowColorImage(imDisp, strName, strAnnotation='', textColor=(0, 0, 255),
                     resizeAmount=None):

    if resizeAmount is not None:
        imDisp = cv2.resize(imDisp.copy(), None, fx=resizeAmount,
                            fy=resizeAmount)

    if len(strAnnotation) > 0:
        cv2.putText(imDisp, strAnnotation, (10, 20), cv2.FONT_HERSHEY_PLAIN,
                    2.0, textColor, thickness=2)

    cv2.imshow(strName, imDisp)


def mplotShowImage(imInput):

    plt.imshow(imInput, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks(())
    plt.yticks(())


def normalizeArray(a):
    return np.single(0.0 + a - a.min()) / (a.max() - a.min())


def AddTextOnImage(imInput, strText, loc=(2, 2), color=255):

    imInputPIL = PIL.Image.fromarray(imInput)
    d = ImageDraw.Draw(imInputPIL)
    d.text(loc, strText, fill=color)
    return np.asarray(imInputPIL)


def AddTextOnVideo(imVideo, strText, loc=(2, 2)):

    imVideoOut = np.zeros_like(imVideo)

    for i in range(imVideo.shape[2]):
        imVideoOut[:, :, i] = AddTextOnImage(imVideo[:, :, i], strText, loc)

    return imVideoOut


def cvShowVideo(imVideo, strWindowName, waitTime=30, resizeAmount=None):

    if not isinstance(imVideo, list):
        imVideo = [imVideo]
        strWindowName = [strWindowName]

    # find max number of frames
    maxFrames = 0

    for vid in range(len(imVideo)):

        if imVideo[vid].shape[-1] > maxFrames:
            maxFrames = imVideo[vid].shape[2]

    # display video
    blnLoop = True
    fid = 0

    while True:

        for vid in range(len(imVideo)):

            curVideoFid = fid % imVideo[vid].shape[2]
            imCur = imVideo[vid][:, :, curVideoFid]

            # resize image if requested
            if resizeAmount:
                imCur = scipy.misc.imresize(imCur, resizeAmount)

            # show image
            cvShowImage(imCur, strWindowName[vid], '%d' % (curVideoFid + 1))

        # look for "esc" key
        k = cv2.waitKey(waitTime) & 0xff

        if blnLoop:

            if k == 27:
                break
            elif k == ord(' '):
                blnLoop = False
            else:
                fid = (fid + 1) % maxFrames

        else:

            if k == 27:  # escape

                break

            elif k == ord(' '):  # space

                blnLoop = True

            elif k == 81:  # left arrow

                fid = (fid - 1) % maxFrames

            elif k == 83:  # right arrow

                fid = (fid + 1) % maxFrames

    for vid in range(len(imVideo)):
        cv2.destroyWindow(strWindowName[vid])


def normalizeArray(a, bounds=None):

    if bounds is None:
        return (0.0 + a - a.min()) / (a.max() - a.min())
    else:
        b = (0.0 + a - bounds[0]) / (bounds[1] - bounds[0])
        b[b < 0] = bounds[0]
        b[b > bounds[1]] = bounds[1]
        return b


def loadVideoFromFile(dataFilePath, sigmaSmooth=None, resizeAmount=None):

    vidseq = cv2.VideoCapture(dataFilePath)

    print vidseq, vidseq.isOpened()

    # print metadata
    metadata = {}

    numFrames = vidseq.get(cv2.CAP_PROP_FRAME_COUNT)
    print '\tFRAME_COUNT = ', numFrames
    metadata['FRAME_COUNT'] = numFrames

    frameHeight = vidseq.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if frameHeight > 0:
        print '\tFRAME HEIGHT = ', frameHeight
        metadata['FRAME_HEIGHT'] = frameHeight

    frameWidth = vidseq.get(cv2.CAP_PROP_FRAME_WIDTH)
    if frameWidth > 0:
        print '\tFRAME WIDTH = ', frameWidth
        metadata['FRAME_WIDTH'] = frameWidth

    fps = vidseq.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        print '\tFPS = ', fps
        metadata['FPS'] = fps

    fmt = vidseq.get(cv2.CAP_PROP_FORMAT)
    if fmt > 0:
        print '\FORMAT = ', fmt
        metadata['FORMAT'] = fmt

    vmode = vidseq.get(cv2.CAP_PROP_MODE)
    if vmode > 0:
        print '\MODE = ', vmode
        metadata['MODE'] = MODE

    # smooth if wanted
    if sigmaSmooth:
        wSmooth = 4 * sigmaSmooth + 1

    print metadata

    # read video frames
    imInput = []

    fid = 0
    prevPercent = 0
    print '\n'

    while True:
        valid_object, frame = vidseq.read()

        if not valid_object:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if resizeAmount:
            frame = scipy.misc.imresize(frame, resizeAmount)

        if sigmaSmooth:
            frame = cv2.GaussianBlur(frame, (wSmooth, wSmooth), 0)

        imInput.append(frame)

        # update progress
        fid += 1
        curPercent = np.floor(100.0 * fid / numFrames)
        if curPercent > prevPercent:
            prevPercent = curPercent
            print '%.2d%%' % curPercent,

    print '\n'

    imInput = np.dstack(imInput)

    vidseq.release()

    return (imInput, metadata)


def writeVideoToFile(imVideo, filename, codec='DIVX', fps=30, isColor=False):

    # start timer
    tStart = time.time()

    # write video
    # fourcc = cv2.FOURCC(*list(codec))    # opencv 2.4
    fourcc = cv2.VideoWriter_fourcc(*list(codec))

    height, width = imVideo.shape[:2]

    writer = cv2.VideoWriter(filename, fourcc, fps=fps,
                             frameSize=(width, height), isColor=isColor)

    print writer.isOpened()

    numFrames = imVideo.shape[-1]

    for fid in range(numFrames):

        if isColor:
            writer.write(imVideo[:, :, :, fid].astype('uint8'))
        else:
            writer.write(imVideo[:, :, fid].astype('uint8'))

    # end timer
    tEnd = time.time()
    print 'Writing video {} took {} seconds'.format(filename, tEnd - tStart)

    # release
    writer.release()


def writeVideoAsTiffStack(imVideo, strFilePrefix):

    # start timer
    tStart = time.time()

    for fid in range(imVideo.shape[2]):
        plt.imsave(strFilePrefix + '.%.3d.tif' % (fid + 1), imVideo[:, :, fid])

    # end timer
    tEnd = time.time()
    print 'Writing video {} took {} seconds'.format(strFilePrefix,
                                                    tEnd - tStart)


def mplotShowMIP(im, axis, xlabel=None, ylabel=None, title=None):

    plt.imshow(im.max(axis))

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)


def convertFromRFtoBMode(imInputRF):

    return np.abs(scipy.signal.hilbert(imInputRF, axis=0))


def normalizeAngles(angleList, angle_range):

    return np.array(
        [angles.normalize(i, angle_range[0], angle_range[1]) for i in
         angleList])


def SaveFigToDisk(saveDir, fileName, saveext=('.png', '.eps'), **kwargs):

    for ext in saveext:
        plt.savefig(os.path.join(saveDir, fileName + ext), **kwargs)


def SaveImageToDisk(im, saveDir, fileName, saveext=('.png',)):
    for ext in saveext:
        plt.imsave(os.path.join(saveDir, fileName + ext), im)


def generateGatedVideoUsingSplineInterp(imInput, numOutFrames, minFrame,
                                        maxFrame, splineOrder):
    tZoom = np.float(numOutFrames) / (maxFrame - minFrame + 1)

    return scipy.ndimage.interpolation.zoom(
        imInput[:, :, minFrame:maxFrame + 1], (1, 1, tZoom), order=splineOrder)


def ncorr(imA, imB):
    imA = (imA - imA.mean()) / imA.std()
    imB = (imB - imB.mean()) / imB.std()

    return np.mean(imA * imB)


def vis_checkerboard(im1, im2):
    im_chk = sitk.CheckerBoard(sitk.GetImageFromArray(im1),
                               sitk.GetImageFromArray(im2))

    return sitk.GetArrayFromImage(im_chk)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with
           RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf