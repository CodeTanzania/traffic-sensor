#ifndef BLOBTRACKER_H
#define BLOBTRACKER_H

#include <chrono>
#include <iostream>
#include <ctime>

#include "patterns/subject.h"
#include "count-topic.h"
#include "blob.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {

            class BlobBuilder; // foward declaration

            // typedefs
            typedef std::array<cv::Point, 2> Line;

            typedef std::vector<cv::Rect> Rects;

            typedef std::vector<cv::Point> Contour;
            typedef std::vector<std::vector<cv::Point>> ContourVec;
            typedef std::vector<std::vector<cv::Point>>::iterator ContourVecIter;
            typedef std::vector<std::vector<cv::Point>>::const_iterator ConstContourVecIter;

            typedef std::vector<Blob> Blobs;
            typedef std::vector<Blob>::iterator BlobsIter;
            typedef std::vector<Blob>::const_iterator BlobsConstIter;

            // definitions
            #define COLOR_YELLOW    cv::Scalar(0, 255, 255)
            #define COLOR_RED       cv::Scalar(0, 0, 255)
            #define COLOR_BLACK     cv::Scalar(0, 0, 0)
            #define COLOR_WHITE     cv::Scalar(255, 255, 255)
            #define COLOR_BLUE      cv::Scalar(255, 0, 0)
            #define COLOR_GREEN     cv::Scalar(0, 255, 0)

            /**
             * @brief final class BlobTracker is the implementation of an algorithm used to track blobs in a stream of video (camera or video file)
             *
             * The class uses filter params that you may wish to customize in order to reduce false positives. Also, you may optinally improve object
             * recognition supported by the GoogLeNet deep learning model to improve the performance by tracking objects of iterest.
             */
            class BlobTracker final : public com::github::kmoz::Subject<CountTopic>
            {
            // blob tracker definition
            public:

                /**
                 * @brief FilterParams gives the parameters that might help to reduce detection of false positives.
                 *
                 * Customization of the parameters is done via a tracker.yaml file located in the res/ folder of this project
                 */
                typedef struct
                {// filter parameters
                    double minArea;
                    double maxArea;
                    double minAspectRatio;
                    double maxAspectRatio;
                    double maxWidth;
                    double minWidth;
                    double maxHeight;
                    double minHeight;
                    double minDiagonal;
                    double maxDiagonal;

                    /// \brief write the filter parameters to the config file
                    void write(cv::FileStorage &fs) const
                    {//write params to tracker.yaml
                        fs << "Params" << "{"
                           << "minArea" << minArea
                           << "maxArea" << maxArea
                           << "minAspectRatio" << minAspectRatio
                           << "maxAspectRatio" << maxAspectRatio
                           << "maxWidth" << maxWidth
                           << "minWidth" << minWidth
                           << "maxHeight" << maxHeight
                           << "minHeight" << minHeight
                           << "minDiagonal" << minDiagonal
                           << "maxDiagonal" << maxDiagonal
                           << "}";
                    } // end write

                    /// \brief read the filter parameters from the config file
                    void read(const cv::FileNode &node)
                    { // read params from tracker.yaml
                        minArea = (double) node["minArea"];
                        maxArea = (double) node["maxArea"];
                        minAspectRatio = (double) node["minAspectRatio"];
                        maxAspectRatio = (double) node["maxAspectRatio"];
                        maxWidth = (double) node["maxWidth"];
                        minWidth = (double) node["minWidth"];
                        maxHeight = (double) node["maxHeight"];
                        minHeight = (double) node["minHeight"];
                        minDiagonal = (double) node["minDiagonal"];
                        maxDiagonal = (double) node["maxDiagonal"];
                    } // end read

                } FilterParams; // end params typedef

                FilterParams params; // define filter params

                // constructors
                /**
                 * @brief construct #BlobTracker to track #{Blob}s crossing through #crossingLine.
                 *
                 * The tracker consumes the information provided in the #configFile in order to filter out
                 * false positives.
                 *
                 * This params are located in the tracker.yaml file in the Params section. They can be
                 * modified to fit the client needs.
                 *
                 * The program needs to restart once the tracker.yaml file is changed.
                 * @param crossingLine an array of two points. When a blob crosses the line formed by these points,
                 * the counting is done.
                 * @see BlobTracker#countCrossingBlobs
                 *
                 * @param configFile the configuration file containing parameters used to configure the tracker
                 * in order to reduce false positives.
                 */
                explicit BlobTracker(const Line &crossingLine, const std::string &configFile);

                /**
                 * @brief construct #BlobTracker to track #{Blob}s crossing through #crossingLine.
                 *
                 * The tracker consume information provided as arguments.
                 * The tracker willl also write the passed configuration to the #configFile.
                 *
                 * @param crossingLine an array of two points. When a blob crosses the line formed by these points,
                 * the counting is done.
                 *
                 * @param configFile the configuration file used to store parameters so that next time they may be remembered
                 *
                 * @param params the filter parameters used to reduce false positive by establishing an hypothesis of
                 * how the blob might be looking in terms of size.
                 *
                 * @param useDnn when set to true, the tracker will use GoogLeNet Model in order to recognize the objects of
                 * interest.
                 */
                explicit BlobTracker(const Line &crossingLine, const std::string &configFile, const FilterParams &params, const bool useDnn = false);

                // public member functions
                /**
                 * @brief update blob movements by calculating the difference between the two frames.
                 *
                 * @param frame0 reference to othe first frame.
                 * @param frame1 reference to the second frame.
                 */
                bool update(cv::Mat &frame0, cv::Mat &frame1);

            private:
                // member variables
                unsigned long mBlobsDirDownCount;       // number of blobs went down across the line
                unsigned long mBlobsDirUpCount;         // number of blobs went up across the line
                unsigned long guid;                     // id of the user.
                Line mCrossingLine;                     // points forming the crossing line
                Blobs mBlobs;                           // a list of detected blobs
                bool mFirstRun;                         // control internals by initializing some resources e.g. FPS
                unsigned long mFramesCount;             // number of frames
                bool mUseDnn;                           // if we should use GoogLeNet model to perform object recognition
                cv::FileNode mConfigFile;               // configuration file
                std::time_t mStartTime;                      // start time. Used to compute fps
                std::time_t mCurrTime;                       // current time. Use to compute fps
                double mFPS;                            // number of frames per second

                // private member functions
                Blobs filter(const Blobs &blobs);                                   // filter blobs that passes the criteria set by params
                Rects filter(const Rects &rects);                                   // filter blobs by using rects
                bool paramsFilter(const Blob &) const;                              // use params to filter blobs
                bool dnnFilter(const Blob &) const;                                 // perform object recognition by using googLeNet model
                void advancePreviousDetections(Blobs &blobs);                       // perform comparison between old and newer detections in order to update or add new detections
                void update(int index, Blob &newBlob);                              // update old detection at the index using newBlob info
                void append(Blob &blob);                                            // add new detection to a list of current blobs
                void removeDormat();                                                // remove old detections that are not active

                // compute the distance between two points
                inline double distance(const cv::Point &p1, const cv::Point &p2)
                { // compute eucledian distance between a pair of points
                    double deltaX = (p2.x - p1.x);
                    double deltaY = (p2.y - p1.y);
                    return std::sqrt((deltaX * deltaX) + (deltaY * deltaY));
                } // end eucledian distance between two points

                // draw blob infos
                void drawBlobInfos(cv::Mat &frame);

                // draw fps
                void drawFPS(cv::Mat &frame);

                // draw crossing line
                void drawCrossingLine(cv::Mat &frame);

                // draw statististics
                void drawStatistics(cv::Mat &frame);

                void countCrossingBlobs();

                // get FPS
                inline double fps()
                { // compute current time FPS
                    double time_diff = std::difftime(std::time(nullptr), mStartTime);
                    if (time_diff == 0.)
                    {
                        return 0.0;
                    }

                    if (mFramesCount == LONG_MAX - 1)
                    {
                        mFramesCount = 0L;
                    }

                    return mFramesCount++ / time_diff;
                } // end fps()

                inline void write(const std::string &configFile) const
                { // save parameters used to configure object detection.

                    // write /override configs
                    cv::FileStorage fs(configFile, cv::FileStorage::WRITE);

                    // if we should use dnn
                    fs << "Options" << "{" << "useDnn" << mUseDnn << "}";

                    // if we should write params
                    if (&params != NULL)
                    {
                        params.write(fs);
                    }

                    // release file storage
                    fs.release();
                } // end write()

                inline void read(const std::string &configFile)
                { // load parameters used to configure object detection.
                    // read configurations
                    cv::FileStorage fs(configFile, cv::FileStorage::READ);

                    // file must exist
                    // CV_Assert(&fs != nullptr);

                    // read params
                    cv::FileNode node = fs["Params"];

                    // if (&node != nullptr)
                    params.read(node);

                    node = fs["Options"];
                    mUseDnn = static_cast<int>(node["useDnn"]) != 0;

                    // should we use dnn for object recognition?
                    if (mUseDnn)
                    {
                        // TODO: initialize the dnn module
                    }

                    // release file storage
                    fs.release();
                } // end read()

                // friends
                friend class BlobBuilder;
            };


        } // end namespace codetanzania
    } // end namespace github
} // end namespace com

#endif // BLOBTRACKER_H
