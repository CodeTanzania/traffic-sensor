#ifndef TRAFFIC_SENSOR_BLOB
#define TRAFFIC_SENSOR_BLOB

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "counter.h"


namespace com {
    namespace github {
    namespace codetanzania {
            class Blob: Counter<Blob> {
            public:
                // constructor
                explicit Blob(const std::vector<cv::Point> &contour);

                unsigned long long id() const { return mId; }

                cv::Rect boundingRect() const {
                    return mCurrBoundingRect;
                }

                std::vector<cv::Point> currentContour() const {
                    return mCurrContour;
                }

                std::vector<cv::Point> centers() const {
                    return mCenterPositions;
                }

                double digonalLength() const {
                    return mCurrDiagonalSize;
                }

                double aspectRatio() const {
                    return mCurrAspectRatio;
                }

                bool active() const {
                    return mActive;
                }

                bool matchFoundOrIsNew() {
                    return mMatchFoundOrIsNew;
                }

                unsigned int currentMissingFramesCount() const {
                    return nMissingFramesCount;
                }

                cv::Point nextPosition() const {
                    return mPredictedNxtPos;
                }

                void setNextPosition(const cv::Point &pos) {
                    mPredictedNxtPos = pos;
                }

                void setId(const unsigned long long id) {
                    mId = id;
                }

                void resetMatchFoundOrIsNew() {
                    mMatchFoundOrIsNew = false;
                }

                void setMatchFoundOrIsNew() {
                    mMatchFoundOrIsNew = true;
                }

                void setCurrContour(const std::vector<cv::Point> &cnt) {
                    mCurrContour = cnt;
                }

                void setCurrBoundingRect(const cv::Rect& rect) {
                    mCurrBoundingRect = rect;
                }

                void addCenterPositions(const cv::Point center) {
                    mCenterPositions.push_back(center);
                }

                void setCurrAspectRatio(const double ratio) {
                    mCurrAspectRatio = ratio;
                }

                void setActive(const bool active) {
                    mActive = active;
                }

                void setDiagonal(const double diag) {
                    mCurrDiagonalSize = diag;
                }

                void incMissingFramesCount() { nMissingFramesCount++; }

                void predictNextPosition();

                void update(const Blob& other) {
                    setMatchFoundOrIsNew();
                    setCurrBoundingRect(other.boundingRect());
                    setCurrContour(other.currentContour());
                    addCenterPositions(other.centers().back());
                    setDiagonal(other.digonalLength());
                    setCurrAspectRatio(other.aspectRatio());
                    setActive(true);
                }

            private:
                unsigned long long mId;                     // id of the blob
                std::vector<cv::Point> mCurrContour;        // current contour
                cv::Rect mCurrBoundingRect;                 // bounding rectangle for the blob
                std::vector<cv::Point> mCenterPositions;    // center positions of the current blobs
                double mCurrDiagonalSize;                   // current diagonal size used to measure drifting of the blob between frames
                double mCurrAspectRatio;                    // the current aspect ration of the blob. used to run comparison algorithm in order to establish  tracking
                bool mMatchFoundOrIsNew;                    // indicate if the blob exists or is new in the scene
                bool mActive;                               // indicate if the blob is currently being tracked.
                unsigned int nMissingFramesCount;           // number of consecutive frames.
                cv::Point mPredictedNxtPos;                 // store a point that estimates the future position fo the blob.
            };
        }
    }
}

#endif    // TRAFFIC_SENSOR_BLOB
