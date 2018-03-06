#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>

#include "arrow-open.h"
#include "blobbuilder.h"
#include "blobtracker.h"
#include "counter.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {

            BlobTracker::BlobTracker(const Line &crossingLine, const std::string &configFile)
            {
                mCrossingLine = crossingLine;
                mFirstRun = true;
                guid = 0;
                mBlobsDirDownCount = 0;
                mBlobsDirUpCount   = 0;
                mFramesCount = 0;
                // the rest member vars are initialized from the `tracker.yaml` file
                read(configFile);
            }

            BlobTracker::BlobTracker(const Line &crossingLine, const std::string &configFile, const FilterParams &params, const bool useDnn)
            {
                this->mCrossingLine = crossingLine;
                this->mFirstRun = true;
                this->guid = 0;
                this->mUseDnn = useDnn;
                this->params = params;
                this->mBlobsDirDownCount = 0;
                this->mBlobsDirUpCount   = 0;
                this->mFramesCount = 0;

                // check if we should use the dnn module
                if (useDnn) {
                    // TODO: use dnn module for object recognition
                }

                write(configFile);
            }

            // filter items in a vector
            Blobs BlobTracker::filter(const Blobs &blobs)
            {
                Blobs result;

                // TODO: Add dnn filter
                std::copy_if(blobs.begin(),
                             blobs.end(),
                             std::back_inserter(result),
                             [&](const Blob &blob){ return this->paramsFilter(blob); }
                );
                return result;
            }

            // filter blobs by using params
            bool BlobTracker::paramsFilter(const Blob &blob) const
            {
                return  (blob.boundingRect().area() > params.minArea)   &&
                        (blob.aspectRatio() > params.minAspectRatio)    &&
                        (blob.aspectRatio() < params.maxAspectRatio)    &&
                        (blob.boundingRect().width > params.minWidth)   &&
                        (blob.boundingRect().height > params.minHeight) &&
                        (blob.digonalLength() > params.minDiagonal)     &&
                        (cv::contourArea(blob.currentContour()) / (double)blob.boundingRect().area() > 0.5);
            }

            // use dnn to filter objects
            bool BlobTracker::dnnFilter(const Blob &) const
            {
                // implementation is required
                CV_Assert(false);
                return false;
            }

            // update blob at the given index.
            void BlobTracker::update(int index, Blob &blob)
            {
                mBlobs.at(index).update(blob);
            }

            // append new blob
            void BlobTracker::append(Blob &blob)
            {
                blob.setMatchFoundOrIsNew();
                mBlobs.push_back(blob);
            }

            // count number of blobs crossing the line
            void BlobTracker::countCrossingBlobs()
            {
                std::for_each(mBlobs.begin(), mBlobs.end(), [&](const Blob &blob)
                {
                   if (blob.active() && blob.centers().size() >= 2)
                   {
                       unsigned prevIndex = blob.centers().size() - 2;
                       unsigned currIndex = blob.centers().size() - 1;

                       cv::Point prevPos = blob.centers().at(prevIndex);
                       cv::Point currPos = blob.centers().at(currIndex);

                       if (prevPos.y < mCrossingLine[0].y && currPos.y > mCrossingLine[0].y)
                       {
                           // moving upwards
                           mBlobsDirDownCount++;
                           CountTopic topic;
                           topic.setCount(CountTopic::CountDirection::COUNT_MOVING_TOWARDS, mBlobsDirDownCount);
                           notify(topic);
                           // emit emitBlobMovedTowardsEvent(mBlobsDirDownCount);
                       }
                       else if (prevPos.y > mCrossingLine[0].y && currPos.y < mCrossingLine[0].y)
                       {
                           // moving down
                           mBlobsDirUpCount++;
                           CountTopic topic;
                           topic.setCount(CountTopic::CountDirection::COUNT_MOVING_AWAY, mBlobsDirUpCount);
                           notify(topic);
                           // emit emitBlobMovedAwayEvent(mBlobsDirUpCount);
                       }
                   }
                });
            }

            void BlobTracker::advancePreviousDetections(Blobs &blobs)
            {
                // run prediction algorithm for each blob
                std::for_each(mBlobs.begin(), mBlobs.end(),
                    [](Blob &blob)
                    {
                        blob.resetMatchFoundOrIsNew();
                        blob.predictNextPosition();
                    });
                // match previous detections
                for (Blob &blob : blobs)
                {

                    int idxCloseEnough = -1;
                    unsigned i;
                    double dstCloseEnough = 100000.0;

                    for (i = 0; i < mBlobs.size(); i++)
                    {
                        if (mBlobs.at(i).active())
                        {
                            double r = distance(blob.centers().back(), mBlobs.at(i).nextPosition());
                            if (r < dstCloseEnough)
                            {
                                dstCloseEnough = r;
                                idxCloseEnough = i;
                            }
                        }
                    }

                    // check if the blob is new or existing. update if necessary
                    if (dstCloseEnough < blob.digonalLength() * 0.5)
                    { // distance cannot exceed half the diagonal
                        update(idxCloseEnough, blob);
                    }
                    else
                    {
                        guid++;
                        blob.setId(guid);
                        append(blob);
                    }
                }

                // stop to track hard to detect blobs
                std::for_each(mBlobs.begin(), mBlobs.end(),
                              [](Blob &blob)
                {
                    if (!blob.matchFoundOrIsNew())
                    {
                        blob.incMissingFramesCount();
                    }

                    if (blob.currentMissingFramesCount() >= 5) {
                        blob.setActive(false);
                    }
                });
            }

            void BlobTracker::drawBlobInfos(cv::Mat &frame)
            {
                for (unsigned int i = 0; i < mBlobs.size(); i++)
                {

                    if (mBlobs.at(i).active())
                    {
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

            void BlobTracker::drawCrossingLine(cv::Mat &frame)
            {
                cv::line(frame, mCrossingLine[0], mCrossingLine[1], COLOR_RED, 2);
            }

            void BlobTracker::drawStatistics(cv::Mat &frame) {

                int width = frame.cols;
                int height = frame.rows;

                cv::Mat roiUpCount = frame(cv::Rect(cv::Point(0, height - 120), cv::Point(width / 2, height)));
                cv::Mat roiDownCount = frame(cv::Rect(cv::Point(width / 2, height - 120), cv::Point(width, height)));

                cv::Mat colorUpCount(roiUpCount.size(), CV_8UC3, COLOR_GREEN);
                cv::Mat colorDownCount(roiDownCount.size(), CV_8UC3, COLOR_BLUE);
                double alpha = 0.3;
                cv::addWeighted(colorUpCount, alpha, roiUpCount, 1 - alpha, 0.0, roiUpCount);
                cv::addWeighted(colorDownCount, alpha, roiDownCount, 1.0 - 0.3, 0.0, roiDownCount);

                int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                double fontScale = (frame.rows * frame.cols) / 300000.0;
                int fontThickness = (int)std::round(fontScale * 1.5);


                cv::Point downTextPos;
                cv::Point upTextPos;

                downTextPos.x = 30;
                downTextPos.y = height - 30;

                upTextPos.x = (width / 2) + 30;
                upTextPos.y = height - 30;

                cv::putText(frame, std::to_string(mBlobsDirUpCount), upTextPos, fontFace, fontScale, COLOR_BLACK, fontThickness);
                cv::putText(frame, std::to_string(mBlobsDirDownCount), downTextPos, fontFace, fontScale, COLOR_BLACK, fontThickness);

                // draw arrows
                int arrowHeight = 30;
                cv::Point arrowUpStart;
                cv::Point arrowUpEnd;
                cv::Point arrowUpLeftVertex;
                cv::Point arrowUpRightVertex;

                arrowUpStart.x = (0.75 * width) ;
                arrowUpStart.y = mCrossingLine[0].y - arrowHeight;
                arrowUpEnd.x = (0.75 * width);
                arrowUpEnd.y = mCrossingLine[0].y + arrowHeight;
                arrowUpLeftVertex.x = arrowUpStart.x - 20;
                arrowUpLeftVertex.y = arrowUpStart.y + 20;
                arrowUpRightVertex.x = arrowUpStart.x + 20;
                arrowUpRightVertex.y = arrowUpStart.y + 20;

                cv::line(frame, arrowUpStart, arrowUpLeftVertex, COLOR_BLUE, 2);
                cv::line(frame, arrowUpStart, arrowUpRightVertex, COLOR_BLUE, 2);
                cv::line(frame, arrowUpStart, arrowUpEnd, COLOR_BLUE, 2);

                cv::Point arrowDownStart;
                cv::Point arrowDownEnd;
                cv::Point arrowDownLeftVertex;
                cv::Point arrowDownRightVertex;

                arrowDownStart.x = (0.25 * width) ;
                arrowDownStart.y = mCrossingLine[0].y - arrowHeight;
                arrowDownEnd.x = (0.25 * width);
                arrowDownEnd.y = mCrossingLine[0].y + arrowHeight;
                arrowDownLeftVertex.x = arrowDownEnd.x - 20;
                arrowDownLeftVertex.y = arrowDownEnd.y - 20;
                arrowDownRightVertex.x = arrowDownEnd.x + 20;
                arrowDownRightVertex.y = arrowDownEnd.y - 20;

                cv::line(frame, arrowDownEnd, arrowDownLeftVertex, COLOR_GREEN, 2);
                cv::line(frame, arrowDownEnd, arrowDownRightVertex, COLOR_GREEN, 2);
                cv::line(frame, arrowDownStart, arrowDownEnd, COLOR_GREEN, 2);
            }

            void BlobTracker::drawFPS(cv::Mat &frame) {
                std::stringstream ss;
                ss << "FPS: " << std::setprecision(5) <<  fps();
                cv::putText(frame, ss.str(), cv::Point(60, 60), CV_FONT_HERSHEY_SIMPLEX, 1. , COLOR_WHITE, 2);
            }

            void BlobTracker::removeDormat()
            {
               std::vector<int> ids;

                for (unsigned int i = 0; i < mBlobs.size(); i++)
                {
                    if (!mBlobs.at(i).active())
                    {
                        ids.push_back(i);
                    }
                }

                for (const int id : ids)
                {
                    mBlobs.erase(mBlobs.begin() + id);
                }
            }

            // update tracker using two frames. We're using
            bool BlobTracker::update(cv::Mat &frame0, cv::Mat &frame1)
            {

                if (!mFirstRun)
                {
                    removeDormat();
                }

                // gray images
                cv::Mat gray0, gray1;

                // convert to single channel if possible
                if (frame0.type() != CV_8UC1)
                {
                    cv::cvtColor(frame0, gray0, CV_BGR2GRAY);
                }
                else
                {
                    gray0 = frame0.clone();
                }

                if (frame1.type() != CV_8UC1)
                {
                    cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
                }
                else
                {
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
                ContourVec contours;
                cv::findContours(imThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                ContourVec convexHulls(contours.size());
                int k = 0;
                for (ConstContourVecIter it = contours.begin(); it != contours.end(); it++)
                {
                    cv::convexHull(*it, convexHulls[k++]);
                }

                cv::Mat frame = frame1.clone();

                Blobs t1Blobs;
                for (ConstContourVecIter it = convexHulls.begin(); it != convexHulls.end(); it++)
                {
                   Blob *blob = BlobBuilder::create(*it, *this);
                   if (blob != nullptr)
                   {
                       t1Blobs.push_back(*blob);
                   }
                }

                // now filter blobs
                Blobs t2Blobs = filter(t1Blobs);
                if (mFirstRun)
                { // if this our first time?
                    // move blobs to the global container
                    std::move(t2Blobs.begin(), t2Blobs.end(), std::back_inserter(mBlobs));
                    // the camera had two frames to begin with
                    mFramesCount += 2;
                    // no need to repeat this operation.
                    mFirstRun = false;
                    // mark starting time
                    mStartTime = std::time(nullptr);
                }
                else
                {
                    // advance previous detections into the current detections. discard old enough/occulsions
                    advancePreviousDetections(t2Blobs);
                }                

                // count number of vehicles
                countCrossingBlobs();

                // mark current time
                mCurrTime = std::time(nullptr);

                // draw frame
                drawBlobInfos(frame);

                // draw crossing line
                drawCrossingLine(frame);

                // draw statistics
                drawStatistics(frame);

                // draw frames count
                drawFPS(frame);

                cv::imshow("Detections", frame);


                return true;
            }
        } // end codetanzania
    } // end github
} // end com

