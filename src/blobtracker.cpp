#include <chrono>
#include <fstream>
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

            // some constants use by SSD_MobileNet
            const size_t inWidth = 300;
            const size_t inHeight = 300;
            const float inScaleFactor = 0.007843f;
            const float meanVal = 127.5;

            // Video writer
            std::string videoFileName = "analysis.avi";
            int fourcc = CV_FOURCC('X', 'V', 'I', 'D');
            int O_FPS = 29;
            cv::VideoWriter vidOut;


            BlobTracker::BlobTracker(const Line &crossingLine1,
                const Line &crossingLine2,
                const Line &crossingLine3,
                const std::string &configFile)
            {
                mUpperCrossingLine = crossingLine1;
                mMiddleCrossingLine = crossingLine2;
                mLowerCrossingLine = crossingLine3;
                mFirstRun = true;
                guid = 0;
                mMiddleLineDownCount = 0;
                mMiddleLineUpCount   = 0;
                mUpperLineUpCount = 0;
                mUpperLineDownCount = 0;
                mLowerLineUpCount = 0;
                mLowerLineDownCount = 0;
                mFramesCount = 0;
                // the rest member vars are initialized from the `tracker.yaml` file
                read(configFile);
            }

            BlobTracker::BlobTracker(const Line &crossingLine1,
                const Line &crossingLine2,
                const Line &crossingLine3,
                const std::string &configFile,
                const FilterParams &params,
                const bool useDnn)
            {
                this->mUpperCrossingLine  = crossingLine1;
                this->mMiddleCrossingLine = crossingLine2;
                this->mLowerCrossingLine = crossingLine3;
                this->mFirstRun = true;
                this->guid = 0;
                this->mUseDnn = useDnn;
                this->params = params;
                this->mMiddleLineDownCount = 0;
                this->mMiddleLineUpCount   = 0;
                this->mUpperLineUpCount = 0;
                this->mUpperLineDownCount = 0;
                this->mLowerLineUpCount = 0;
                mLowerLineDownCount = 0;
                this->mFramesCount = 0;

                // check if we should use the dnn module
                if (useDnn) {
                    // TODO: use dnn module for object recognition
                }

                write(configFile);
            }

            // load the class labels
            bool BlobTracker::loadClassLabels(const std::string &filename)
            {
                // open class labels
                std::ifstream fp(filename);
                if (!fp.is_open())
                {
                    std::cerr << "file not found. path to file: `" << filename << "`" << std::endl;
                    // std::exit(-1);
                    return false;
                }

                // store line in loop iterations
                std::string line;

                while(!fp.eof())
                {
                    std::getline(fp, line);
                    if (line.length())
                    {
                        mLabels.emplace_back(line.substr(line.find(' ') + 1));
                    }
                }

                // close the file after
                fp.close();

                // return
                return true;
            }

            bool BlobTracker::initRecognition(
                    const std::string &labelsFile,
                    const std::string &protoTextFile,
                    const std::string &modelBinFile)
            {
                // load class labels
                if(!loadClassLabels(labelsFile))
                {
                    return false;
                }

                // initialize the neural network
                mNet = cv::dnn::readNetFromCaffe(protoTextFile, modelBinFile);

                if (mNet.empty())
                {
                    std::cerr << "Cannot create caffe.googlenet model from proto txt: "
                              << protoTextFile
                              << " and model binary "
                              << modelBinFile
                              << std::endl;
                    // std::exit(-1);
                    return false;
                }
                else
                {
                    std::cout << "Initialed Recognizer" << std::endl;
                }

                return true;
            }

            // filter items in a vector
            Blobs BlobTracker::filter(const cv::Mat &frame, const Blobs &blobs)
            {
                Blobs result;

                /* for (int i = 0; blobs.size(); i++)
                {
                    Blob blob = blobs[i];
                    if (mUseDnn && dnnFilter(blob, frame))
                    {
                        result.push_back(blob);
                    }
                    else if (paramsFilter(blob))
                    {
                        result.push_back(blob);
                    }
                }*/

                std::copy_if(blobs.begin(),
                             blobs.end(),
                             std::back_inserter(result),
                             [&](const Blob &blob)
                {
                     if (this->mUseDnn)
                     {
                         if (this->paramsFilter(blob))
                         {
                             return this->dnnFilter(blob, frame);
                         }
                         return false;
                     }
                     return this->paramsFilter(blob);
                });

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
            bool BlobTracker::dnnFilter(const Blob &blob, const cv::Mat &frame)
            {
                try
                {
                    cv::Rect roi = blob.boundingRect();
                    cv::Mat img = frame(roi);
                    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
                    cv::Mat inputBlob = cv::dnn::blobFromImage(img, inScaleFactor, cv::Size(inWidth, inHeight), meanVal);
                    cv::imshow("sample", img);
                    mNet.setInput(inputBlob, "data");
                    cv::Mat detection = mNet.forward("detection_out");
                    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
                    for (int i = 0; i < detectionMat.rows; i++)
                    {
                        int classId = (int)detectionMat.at<float>(i, 1);
                        if (classId == 7 || classId == 8)
                        {
                            float confidence = detectionMat.at<float>(i, 2);
                            if (confidence >= 0.2)
                            {
                                return true;
                            }
                        }
                    }
                    return true;
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << e.what() << std::endl;
                }
                return false;
            }

            // get most probable class
            void BlobTracker::getMaxClass(const cv::Mat &probBlob, int &classId, double &classProb)
            {
                cv::Mat probMat = probBlob.reshape(1, 1);
                cv::Point classNumber;

                cv::minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
                classId = classNumber.x;
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

                       // begin --- conditions to check if the blob has crossed one of the three lines
                       bool goingDownThroughRegion1 = prevPos.y < mUpperCrossingLine[0].y &&
                                currPos.y > mUpperCrossingLine[0].y &&
                                currPos.y < mMiddleCrossingLine[0].y;

                       bool goingUpThroughRegion1 = prevPos.y > mUpperCrossingLine[0].y &&
                               prevPos.y < mMiddleCrossingLine[0].y &&
                               currPos.y < mUpperCrossingLine[0].y;

                       bool goingDownThroughRegion2 = prevPos.y < mMiddleCrossingLine[0].y &&
                               currPos.y > mMiddleCrossingLine[0].y &&
                               currPos.y < mLowerCrossingLine[0].y;

                       bool goingUpThroughRegion2 = prevPos.y > mMiddleCrossingLine[0].y &&
                               currPos.y < mMiddleCrossingLine[0].y &&
                               currPos.y > mUpperCrossingLine[0].y;

                       bool goingDownThroughRegion3 = prevPos.y < mLowerCrossingLine[0].y &&
                               prevPos.y > mMiddleCrossingLine[0].y &&
                               currPos.y > mLowerCrossingLine[0].y;

                       bool goingUpThroughRegion3 = prevPos.y > mLowerCrossingLine[0].y &&
                               currPos.y < mLowerCrossingLine[0].y &&
                               currPos.y > mMiddleCrossingLine[0].y;

                       // end -- condition to check if the blob has crossed one of the three lines.

                       CountTopic* topic = NULL;

                       if (goingDownThroughRegion1) {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::UPPER_COUNT_MOVING_TOWARDS,
                                           ++mUpperLineDownCount);
                       }

                       if (goingUpThroughRegion1) {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::UPPER_COUNT_MOVING_AWAY,
                                           ++mUpperLineUpCount);
                       }

                       if (goingDownThroughRegion2)
                       {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::MIDDLE_COUNT_MOVING_TOWARDS,
                                          ++mMiddleLineDownCount);
                       }

                       if (goingUpThroughRegion2)
                       {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::MIDDLE_COUNT_MOVING_AWAY,
                                          ++mMiddleLineUpCount);
                       }

                       if (goingDownThroughRegion3)
                       {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::DOWN_COUNT_MOVING_TOWARDS,
                                           ++mLowerLineDownCount);
                       }

                       if (goingUpThroughRegion3)
                       {
                           topic = new CountTopic();
                           topic->setCount(CountTopic::CountDirection::DOWN_COUNT_MOVING_AWAY,
                                           ++mLowerLineUpCount);
                       }

                       // check if topic was initialized and notify listeners.
                       if (topic != NULL)
                       {
                           notify(*topic);
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
                        cv::Point bottomRight = cv::Point(blob.boundingRect().x +
                                                          blob.boundingRect().width,
                                                          blob.boundingRect().y +
                                                          blob.boundingRect().y);
                        cv::rectangle(frame, topLeft - cv::Point(1, intFilledHeight),
                                      cv::Point(bottomRight.x, blob.boundingRect().y),
                                      cv::Scalar(0, 255, 255), CV_FILLED);
                        cv::putText(frame, std::to_string(blob.id()),
                                    topLeft - cv::Point(-5, 5),intFontFace, fontScale,
                                    cv::Scalar(0, 0, 0), intFontThickness);
                    }
                }
            }

            void BlobTracker::drawCrossingLines(cv::Mat &frame)
            {
                cv::line(frame, mUpperCrossingLine[0], mUpperCrossingLine[1], COLOR_GREEN, 2);
                cv::line(frame, mMiddleCrossingLine[0], mMiddleCrossingLine[1], COLOR_RED, 2);
                cv::line(frame, mLowerCrossingLine[0], mLowerCrossingLine[1], COLOR_BLUE, 2);
            }

            void BlobTracker::drawStatistics(cv::Mat &frame) {

                int width = frame.cols;
                int height = frame.rows;

                cv::Mat roiGreen = frame(cv::Rect(cv::Point(0, height - 120), cv::Point(width / 3, height)));
                cv::Mat roiRed = frame(cv::Rect(cv::Point(width / 3, height - 120), cv::Point(2 * width / 3, height)));
                cv::Mat roiBlue = frame(cv::Rect(cv::Point(2 * width / 3, height - 120), cv::Point(width, height)));

                cv::Mat colorGreen(roiGreen.size(), CV_8UC3, COLOR_GREEN);
                cv::Mat colorRed(roiRed.size(), CV_8UC3, COLOR_RED);
                cv::Mat colorBlue(roiBlue.size(), CV_8UC3, COLOR_BLUE);
                double alpha = 0.3;
                cv::addWeighted(colorGreen, alpha, roiGreen, 1 - alpha, 0.0, roiGreen);
                cv::addWeighted(colorRed, alpha, roiRed, 1.0 - 0.3, 0.0, roiRed);
                cv::addWeighted(colorBlue, alpha, roiBlue, 1 - alpha, 0.0, roiBlue);

                int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                double fontScale = (frame.rows * frame.cols) / 900000.0;
                int fontThickness = (int)std::round(fontScale * 1.5);


                cv::Point redTextDown;
                cv::Point redTextUp;

                redTextDown.x = (width / 3) + 90;
                redTextDown.y = height - 45;

                redTextUp.x = (2 * width / 3) - 200;
                redTextUp.y = height - 45;

                cv::putText(frame, "U: " + std::to_string(mMiddleLineUpCount),
                            redTextUp, fontFace, fontScale, COLOR_BLACK, fontThickness);
                cv::putText(frame, "D: " + std::to_string(mMiddleLineDownCount),
                            redTextDown, fontFace, fontScale, COLOR_BLACK, fontThickness);

                cv::Point greenTextDown;
                cv::Point greenTextUp;

                greenTextDown.x = 90;
                greenTextDown.y = height - 45;
                greenTextUp.x = (width / 3) - 200;
                greenTextUp.y = height - 45;

                cv::putText(frame, "U: " + std::to_string(mUpperLineUpCount),
                            greenTextUp, fontFace, fontScale, COLOR_BLACK, fontThickness);
                cv::putText(frame, "D: " + std::to_string(mUpperLineDownCount),
                            greenTextDown, fontFace, fontScale, COLOR_BLACK, fontThickness);

                cv::Point blueTextDown;
                cv::Point blueTextUp;

                blueTextDown.x = (2 * width / 3) + 90;
                blueTextDown.y = height - 45;
                blueTextUp.x = width - 200;
                blueTextUp.y = height - 45;

                cv::putText(frame, "U: " + std::to_string(mLowerLineUpCount),
                            blueTextUp, fontFace, fontScale, COLOR_BLACK, fontThickness);
                cv::putText(frame, "D: " + std::to_string(mLowerLineDownCount),
                            blueTextDown, fontFace, fontScale, COLOR_BLACK, fontThickness);


                // draw arrows
//                int arrowHeight = 30;
//                cv::Point arrowUpStart;
//                cv::Point arrowUpEnd;
//                cv::Point arrowUpLeftVertex;
//                cv::Point arrowUpRightVertex;

//                arrowUpStart.x = (0.75 * width) ;
//                arrowUpStart.y = mMiddleCrossingLine[0].y - arrowHeight;
//                arrowUpEnd.x = (0.75 * width);
//                arrowUpEnd.y = mMiddleCrossingLine[0].y + arrowHeight;
//                arrowUpLeftVertex.x = arrowUpStart.x - 20;
//                arrowUpLeftVertex.y = arrowUpStart.y + 20;
//                arrowUpRightVertex.x = arrowUpStart.x + 20;
//                arrowUpRightVertex.y = arrowUpStart.y + 20;

//                cv::line(frame, arrowUpStart, arrowUpLeftVertex, COLOR_BLUE, 2);
//                cv::line(frame, arrowUpStart, arrowUpRightVertex, COLOR_BLUE, 2);
//                cv::line(frame, arrowUpStart, arrowUpEnd, COLOR_BLUE, 2);

//                cv::Point arrowDownStart;
//                cv::Point arrowDownEnd;
//                cv::Point arrowDownLeftVertex;
//                cv::Point arrowDownRightVertex;

//                arrowDownStart.x = (0.25 * width) ;
//                arrowDownStart.y = mMiddleCrossingLine[0].y - arrowHeight;
//                arrowDownEnd.x = (0.25 * width);
//                arrowDownEnd.y = mMiddleCrossingLine[0].y + arrowHeight;
//                arrowDownLeftVertex.x = arrowDownEnd.x - 20;
//                arrowDownLeftVertex.y = arrowDownEnd.y - 20;
//                arrowDownRightVertex.x = arrowDownEnd.x + 20;
//                arrowDownRightVertex.y = arrowDownEnd.y - 20;

//                cv::line(frame, arrowDownEnd, arrowDownLeftVertex, COLOR_GREEN, 2);
//                cv::line(frame, arrowDownEnd, arrowDownRightVertex, COLOR_GREEN, 2);
//                cv::line(frame, arrowDownStart, arrowDownEnd, COLOR_GREEN, 2);
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
                Blobs t2Blobs = filter(frame, t1Blobs);
                if (mFirstRun)
                { // if this our first time?
                    // move blobs to the global container
                    std::move(t2Blobs.begin(), t2Blobs.end(), std::back_inserter(mBlobs));
                    // the camera had two frames to begin with
                    mFramesCount += 2;
                    // init video writer
                    vidOut.open(videoFileName, fourcc, O_FPS, cv::Size(frame.cols, frame.rows));
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
                drawCrossingLines(frame);

                // draw statistics
                drawStatistics(frame);

                // draw frames count
                drawFPS(frame);

                cv::imshow("Detections", frame);

                // write video file
                // if (vidOut.isOpened())
                // {
                //     vidOut << frame;
                // }


                return true;
            }
        } // end codetanzania
    } // end github
} // end com

