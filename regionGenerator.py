import cv2
import numpy as np
import threading
from segment import segment

def generateProposals(original,mask,count,imagePieces,row=0,col=0):

    #print(count)

    width = mask.shape[1]
    height = mask.shape[0]

    sumRows = np.sum(mask, axis=1)
    sumCols = np.sum(mask, axis=0)
    maxRowIdx = np.argmax(sumRows)
    maxRowSum = sumRows[maxRowIdx]
    maxColIdx = np.argmax(sumCols)
    maxColSum = sumCols[maxColIdx]

    minCol = maxColIdx
    maxCol = maxColIdx
    minRow = maxRowIdx
    maxRow = height - 1
    avgRow = np.average(sumRows)
    avgCol = np.average(sumCols)

    while True:
        change = False
        if(minCol > 0):
            if(sumCols[minCol - 1] > avgCol/(7-count)):
                minCol = minCol - 1
                change = True
        if(maxCol < width-1):
            if(sumCols[maxCol + 1] > avgCol/(7-count)):
                maxCol = maxCol + 1
                change = True
        if not change:
            break

    while True:
        change = False
        if(minRow > 0):
            if(sumRows[minRow - 1] > avgRow/(7-count)):
                minRow = minRow - 1
                change = True
        if not change:
            break

    mask = mask[minRow:maxRow,minCol:maxCol]
    imagePiece = original[minRow:maxRow,minCol:maxCol]
    if(maxCol - minCol > 0 and maxRow - minRow > 0):
        newSegment = segment(imagePiece,(minRow+row),(minCol+col))
        imagePieces.append(newSegment)
    count = count + 1
    #cv2.imshow("output", cv2.cvtColor(imagePiece, cv2.COLOR_HSV2BGR))

    if count == 3:
        width = mask.shape[1]
        height = mask.shape[0]
        if(height > 2 and width > 0):
            top = imagePiece[0:int(height/2),0:width]
            mid = imagePiece[int(height/4):int(3*height/4),0:width]
            bot = imagePiece[int(height/2):height,0:width]
            topSegment = segment(top,minRow+row,minCol+col)
            midSegment = segment(mid,minRow+row+height/4,minCol+col)
            botSegment = segment(mid,minRow+row+height/2,minCol+col)
            imagePieces.append(topSegment)
            imagePieces.append(midSegment)
            imagePieces.append(botSegment)

        return imagePieces
    else:
        width = mask.shape[1]
        height = mask.shape[0]

        left = mask[0:height,0:int(width/2)]
        leftImage = imagePiece[0:height,0:int(width/2)]
        middle = mask[0:height,int(width/4):int(3*width/4)]
        midImage = imagePiece[0:height,int(width/4):int(3*width/4)]
        right = mask[0:height,int(width/2):width]
        rightImage = imagePiece[0:height,int(width/2):width]

        leftThread = threading.Thread(target=generateProposals, args=(leftImage,left,count,imagePieces,row+minRow,col+minCol))
        leftThread.start()
        midThread = threading.Thread(target=generateProposals, args=(midImage,middle,count,imagePieces,row+minRow,col+minCol+int(width/4)))
        midThread.start()
        rightThread = threading.Thread(target=generateProposals, args=(rightImage,right,count,imagePieces,row+minRow,col+minCol+int(width/2)))
        rightThread.start()
        leftThread.join()
        midThread.join()
        rightThread.join()


        #leftIndex = generateProposals(leftImage,left,count,imagePieces)
        #midIndex = generateProposals(midImage,middle,count,imagePieces)
        #rightIndex = generateProposals(rightImage,right,count,imagePieces)

        #for j in range(len(leftIndex)):
        #    imagePieces.append(leftIndex[j])
        #    imagePieces.append(midIndex[j])
        #    imagePieces.append(rightIndex[j])

        return imagePieces
