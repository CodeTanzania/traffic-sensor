#ifndef BLOBTRACKER_H
#define BLOBTRACKER_H

#include <iostream>

#include "blob.h"

namespace com {
    namespace github {
        namespace codetanzania {

            class BlobBuilder;

            class BlobTracker final {
            public:

                struct FilterParams {
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

                    void write(cv::FileStorage &fs) const {
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
                    }

                    void read(const cv::FileNode &node) {
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
                    }

                } params;

                // constructors
                explicit BlobTracker(const std::array<cv::Point, 2> &crossingLine, const std::string &configFile);
                explicit BlobTracker(const std::array<cv::Point, 2> &crossingLine, const std::string &configFile, const FilterParams &params, const bool useDnn = false);

                // public member functions
                bool update(cv::Mat&, cv::Mat&);

            private:
                // member variables
                unsigned long guid;
                std::array<cv::Point, 2> mCrossingLine;
                std::vector<Blob> mBlobs;
                bool mFirstRun;
                bool mFramesCount;
                bool mUseDnn;
                cv::FileNode mConfigFile;

                // private member functions
                std::vector<Blob> filter(const std::vector<Blob> &blobs);
                std::vector<cv::Rect> filter(const std::vector<cv::Rect> &rects);
                bool paramsFilter(const Blob &) const;      // use params to filter blobs
                bool dnnFilter(const Blob &) const;         // use dnn to filter blobs
                void advancePreviousDetections(std::vector<Blob> &blobs);
                void update(int index, Blob &blob);
                void append(Blob &blob);
                void removeDormat();

                // compute the distance between two points
                double distance(const cv::Point &p1, const cv::Point &p2) {
                    double deltaX = (p2.x - p1.x);
                    double deltaY = (p2.y - p1.y);
                    return std::sqrt((deltaX * deltaX) + (deltaY * deltaY));
                }

                // draw blob infos
                void drawBlobInfos(cv::Mat &frame);

                void write(const std::string &configFile) const { // save parameters used to configure object detection.

                    // write /override configs
                    cv::FileStorage fs(configFile, cv::FileStorage::WRITE);

                    // if we should use dnn
                    fs << "Options" << "{" << "useDnn" << mUseDnn << "}";

                    // if we should write params
                    if (&params != NULL) {
                        params.write(fs);
                    }

                    // release file storage
                    fs.release();
                }

                void read(const std::string &configFile) { // load parameters used to configure object detection.
                    // read configurations
                    cv::FileStorage fs(configFile, cv::FileStorage::READ);

                    // file must exist
                    CV_Assert(&fs != NULL);

                    // read params
                    cv::FileNode node = fs["Params"];

                    if (&node != NULL)
                        params.read(node);

                    node = fs["Options"];
                    mUseDnn = static_cast<int>(node["useDnn"]) != 0;

                    // should we use dnn for object recognition?
                    if (mUseDnn) {
                        // TODO: initialize the dnn module
                    }

                    // release file storage
                    fs.release();
                }
                // friends
                friend class BlobBuilder;
            };


        } // end namespace codetanzania
    } // end namespace github
} // end namespace com

#endif // BLOBTRACKER_H
