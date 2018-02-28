#include "blobbuilder.h"
#include "blobtracker.h"
#include "counter.h"

namespace com{
    namespace github {
        namespace codetanzania {

            BlobTracker::BlobTracker(
                    const std::array<cv::Point, 2> &crossingLine,
                    const std::string &configFile)
            {
                mCrossingLine = crossingLine;
                mFirstRun = true;
                guid = 0;


                // the rest member vars are initialized from the `tracker.yaml` file
                read(configFile);
            }

            BlobTracker::BlobTracker(
                    const std::array<cv::Point, 2> &crossingLine,
                    const std::string &configFile,
                    const FilterParams &params,
                    const bool useDnn)
            {
                this->mCrossingLine = crossingLine;
                this->mFirstRun = true;
                this->guid = 0;
                this->mUseDnn = useDnn;
                this->params = params;

                // check if we should use the dnn module
                if (useDnn) {
                    // TODO: use dnn module for object recognition
                }

                write(configFile);
            }

            // filter items in a vector
            std::vector<Blob> BlobTracker::filter(const std::vector<Blob> &blobs) {
                std::vector<Blob> result;

                // TODO: Add dnn filter
                std::copy_if(blobs.begin(),
                             blobs.end(),
                             std::back_inserter(result),
                             [&](const Blob &blob){ return this->paramsFilter(blob); }
                );
                return result;
            }

            // filter blobs by using params
            bool BlobTracker::paramsFilter(const Blob &blob) const {
                return  (blob.boundingRect().area() > params.minArea)   &&
                        (blob.aspectRatio() > params.minAspectRatio)    &&
                        (blob.aspectRatio() < params.maxAspectRatio)    &&
                        (blob.boundingRect().width > params.minWidth)   &&
                        (blob.boundingRect().height > params.minHeight) &&
                        (blob.digonalLength() > params.minDiagonal)     &&
                        (cv::contourArea(blob.currentContour()) / (double)blob.boundingRect().area() > 0.5);
            }

            // use dnn to filter objects
            bool BlobTracker::dnnFilter(const Blob &) const {
                // implementation is required
                CV_Assert(false);
            }

            // update blob at the given index.
            void BlobTracker::update(int index, Blob &blob) {
                mBlobs.at(index).update(blob);
            }

            // append new blob
            void BlobTracker::append(Blob &blob) {
                blob.setMatchFoundOrIsNew();
                mBlobs.push_back(blob);
            }

            void BlobTracker::advancePreviousDetections(std::vector<Blob> &blobs) {
                // run prediction algorithm for each blob
                std::for_each(mBlobs.begin(), mBlobs.end(),
                    [](Blob &blob) {
                        blob.resetMatchFoundOrIsNew();
                        blob.predictNextPosition();
                    });
                // match previous detections
                for (Blob &blob : blobs) {

                    int idxCloseEnough = -1, i;
                    double dstCloseEnough = 100000.0;

                    for (i = 0; i < mBlobs.size(); i++) {
                        if (mBlobs.at(i).active()) {
                            double r = distance(blob.centers().back(), mBlobs.at(i).nextPosition());
                            if (r < dstCloseEnough) {
                                dstCloseEnough = r;
                                idxCloseEnough = i;
                            }
                        }
                    }

                    // check if the blob is new or existing. update if necessary
                    if (dstCloseEnough < blob.digonalLength() * 0.5) { // distance cannot exceed half the diagonal
                        update(idxCloseEnough, blob);
                    } else {
                        guid++;
                        blob.setId(guid);
                        append(blob);
                    }
                }

                // stop to track hard to detect blobs
                std::for_each(mBlobs.begin(), mBlobs.end(),
                              [](Blob &blob) {
                    if (!blob.matchFoundOrIsNew()) {
                        blob.incMissingFramesCount();
                    }

                    if (blob.currentMissingFramesCount() >= 5) {
                        blob.setActive(false);
                    }
                });
            }

            void BlobTracker::drawBlobInfos(cv::Mat &frame) {
                for (unsigned int i = 0; i < mBlobs.size(); i++) {

                    if (mBlobs.at(i).active()) {

                        Blob blob = mBlobs.at(i);

                        cv::rectangle(frame, blob.boundingRect(), cv::Scalar(0, 255, 255), 2);
                        int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
                        double fontScale = blob.digonalLength() / 120;
                        int intFontThickness = (int)std::round(fontScale * 1.0);
                        int intFilledHeight = (int)::std::round(fontScale * 40.0);
                        cv::Point topLeft = cv::Point(blob.boundingRect().x, blob.boundingRect().y);
                        cv::Point bottomRight = cv::Point(blob.boundingRect().x + blob.boundingRect().width, blob.boundingRect().y + blob.boundingRect().y);
                        cv::rectangle(frame, topLeft - cv::Point(1, intFilledHeight), cv::Point(bottomRight.x, blob.boundingRect().y), cv::Scalar(0, 255, 255), CV_FILLED);
                        cv::putText(frame, std::to_string(blob.id()), topLeft - cv::Point(-5, 5), intFontFace, fontScale, cv::Scalar(0, 0, 0), intFontThickness);
                    }
                }
            }

            void BlobTracker::removeDormat() {

               std::vector<int> ids;

                for (int i = 0; i < mBlobs.size(); i++) {
                    if (!mBlobs.at(i).active()) {
                        ids.push_back(i);
                    }
                }

                for (const int id : ids) {
                    mBlobs.erase(mBlobs.begin() + id);
                }
            }

            // update tracker using two frames. We're using
            bool BlobTracker::update(cv::Mat &frame0, cv::Mat &frame1) {

                if (!mFirstRun) {
                    removeDormat();
                }

                cv::Mat gray0, gray1;

                // convert to single channel if possible
                if (frame0.type() != CV_8UC1) {
                    cv::cvtColor(frame0, gray0, CV_BGR2GRAY);
                } else {
                    gray0 = frame0.clone();
                }

                if (frame1.type() != CV_8UC1) {
                    cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
                } else {
                    gray1 = frame1.clone();
                }

                // enhance image structures at different scales
                cv::GaussianBlur(gray0, gray0, cv::Size(5, 5), 0);
                cv::GaussianBlur(gray1, gray1, cv::Size(5, 5), 0);

                // calculate difference between first and video frame in order to detect moving objects
                cv::Mat imDiff, imThresh;
                cv::absdiff(gray1, gray0, imDiff);

                // binarize the image
                cv::threshold(imDiff, imThresh, 30, 255.0, CV_THRESH_BINARY);

                cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                cv::dilate(imThresh, imThresh, structuringElement5x5);
                cv::dilate(imThresh, imThresh, structuringElement5x5);
                cv::erode(imThresh, imThresh, structuringElement5x5);

                // find contours arround the blobs
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(imThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                std::vector<std::vector<cv::Point> > convexHulls(contours.size());
                int k = 0;
                for (std::vector<std::vector<cv::Point>>::const_iterator it = contours.begin(); it != contours.end(); it++) {
                    cv::convexHull(*it, convexHulls[k++]);
                }

                cv::Mat frame = frame1.clone();

                std::vector<Blob> t1Blobs;
                for (std::vector<std::vector<cv::Point>>::const_iterator it = convexHulls.begin(); it != convexHulls.end(); it++) {
                   Blob *blob = BlobBuilder::create(*it, *this);
                   if (blob != nullptr) {
                       t1Blobs.push_back(*blob);
                   }
                }

                // now filter blobs
                std::vector<Blob> t2Blobs = filter(t1Blobs);
                if (mFirstRun) { // if this our first time?
                    // move blobs to the global container
                    std::move(t2Blobs.begin(), t2Blobs.end(), std::back_inserter(mBlobs));
                    // the camera had two frames to begin with
                    mFramesCount += 2;
                    // no need to repeat this operation.
                    mFirstRun = false;
                } else {
                    // advance previous detections into the current detections. discard old enough/occulsions
                    advancePreviousDetections(t2Blobs);
                }

                // draw frame
                drawBlobInfos(frame);

                cv::imshow("Detections", frame);

                return true;
            }
        }
    }
}
