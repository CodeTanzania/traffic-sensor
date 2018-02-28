#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "blob.h"
#include "blobtracker.h"

using com::github::codetanzania::Blob;
using com::github::codetanzania::BlobTracker;

int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat frame0;
    cv::Mat frame1;

    std::array<cv::Point, 2> crossingLine;

    capVideo.open("CarsDrivingUnderBridge.mp4");

    if (!capVideo.isOpened()) {
        std::cerr << "error reading video file" << std::endl << std::endl;
        return(-1);
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cerr << "error: video file must have at least two frames" << std::endl;
        return(-1);
    }

    capVideo.read(frame0);
    capVideo.read(frame1);

    int pos = (int)std::round((double)frame0.rows * 0.35);

    crossingLine[0].x = 0;
    crossingLine[0].y = pos;

    crossingLine[1].x = frame0.cols - 1;
    crossingLine[1].y = pos;

    char chCheckForEscKey = 0;

    int frameCount = 2;

    BlobTracker::FilterParams params;

    std::string configFile = "tracker.yaml";


    BlobTracker bt(crossingLine, configFile);

    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        bt.update(frame0, frame1);


        frame0 = frame1.clone(); // move frame 0 up to where frame 1 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(frame1); // read frame 1
        } else {
            std::cout << "end of video\n";
            cv::waitKey(0);
                break;
        }

        frameCount++;
        chCheckForEscKey = cv::waitKey(10);
    }

    return (0);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
//bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
//    bool blnAtLeastOneBlobCrossedTheLine = false;

//    for (auto blob : blobs) {

//        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
//            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
//            int currFrameIndex = (int)blob.centerPositions.size() - 1;

//            if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
//                carCount++;
//                blnAtLeastOneBlobCrossedTheLine = true;
//            }
//        }

//    }

//    return blnAtLeastOneBlobCrossedTheLine;
//}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

//    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
//    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
//    int intFontThickness = (int)std::round(dblFontScale * 1.5);

//    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

//    cv::Point ptTextBottomLeftPosition;

//    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
//    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

//    cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

//}
