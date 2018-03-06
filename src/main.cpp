#include <iostream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "blob.h"
#include "blobtracker.h"
#include "logging-observer.h"

using com::github::codetanzania::Blob;
using com::github::codetanzania::BlobTracker;
using com::github::codetanzania::LoggingObserver;

using cv::Mat;
using cv::Point;
using cv::VideoCapture;
using cv::waitKey;

using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::round;
using std::string;

int main(void)
{

    VideoCapture capVideo;

    Mat frame0;
    Mat frame1;

    array<cv::Point, 2> crossingLine;

    capVideo.open("../res/CarsDrivingUnderBridge.mp4");

    if (!capVideo.isOpened())
    {
        cerr << "error reading video file" << std::endl << std::endl;
        return(-1);
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2)
    {
        cerr << "error: video file must have at least two frames" << std::endl;
        return(-1);
    }

    capVideo.read(frame0);
    capVideo.read(frame1);

    int pos = (int)round((double)frame0.rows * 0.50);

    crossingLine[0].x = 0;
    crossingLine[0].y = pos;

    crossingLine[1].x = frame0.cols - 1;
    crossingLine[1].y = pos;

    char escapeKeyCode = 0;

    // BlobTracker::FilterParams params;

    string configFile = "../res/tracker.yaml";
    BlobTracker bt(crossingLine, configFile);

    LoggingObserver observer;
    bt.attach(&observer);

    while (capVideo.isOpened() && escapeKeyCode != 27)
    {

        bt.update(frame0, frame1);


        frame0 = frame1.clone(); // move frame 0 up to where frame 1 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT))
        {
            capVideo.read(frame1); // read frame 1
        }
        else
        {
            cout << "end of video\n";
            waitKey(0);
                break;
        }

        escapeKeyCode = waitKey(10);
    }

    return (0);
}
