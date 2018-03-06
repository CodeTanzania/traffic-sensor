#include "blob.h"

namespace com {
    namespace github {
        namespace codetanzania {

            Blob::Blob(const std::vector<cv::Point> &contour): Counter() {

                mCurrContour = contour;

                mCurrBoundingRect = cv::boundingRect(mCurrContour);

                cv::Point currentCenter;

                currentCenter.x = (mCurrBoundingRect.x + mCurrBoundingRect.x + mCurrBoundingRect.width) / 2;
                currentCenter.y = (mCurrBoundingRect.y + mCurrBoundingRect.y + mCurrBoundingRect.height) / 2;

                mCenterPositions.push_back(currentCenter);

                mCurrDiagonalSize = sqrt(pow(mCurrBoundingRect.width, 2) + pow(mCurrBoundingRect.height, 2));

                mCurrAspectRatio = (float)mCurrBoundingRect.width / (float)mCurrBoundingRect.height;

                mActive = true;

                mMatchFoundOrIsNew = true;

                nMissingFramesCount = 0;
            }

            void Blob::predictNextPosition() {

                int numPositions = (int)mCenterPositions.size();

                if (numPositions == 1) {

                    mPredictedNxtPos.x = mCenterPositions.back().x;
                    mPredictedNxtPos.y = mCenterPositions.back().y;

                } else if (numPositions == 2) {

                    int deltaX = mCenterPositions[1].x - mCenterPositions[0].x;
                    int deltaY = mCenterPositions[1].y - mCenterPositions[0].y;

                    mPredictedNxtPos.x = mCenterPositions.back().x + deltaX;
                    mPredictedNxtPos.y = mCenterPositions.back().y + deltaY;

                } else if (numPositions == 3) {

                    int sumOfXChanges = ((mCenterPositions[2].x - mCenterPositions[1].x) * 2) +
                        ((mCenterPositions[1].x - mCenterPositions[0].x) * 1);

                    int deltaX = (int)std::round((float)sumOfXChanges / 3.0);

                    int sumOfYChanges = ((mCenterPositions[2].y - mCenterPositions[1].y) * 2) +
                        ((mCenterPositions[1].y - mCenterPositions[0].y) * 1);

                    int deltaY = (int)std::round((float)sumOfYChanges / 3.0);

                    mPredictedNxtPos.x = mCenterPositions.back().x + deltaX;
                    mPredictedNxtPos.y = mCenterPositions.back().y + deltaY;

                } else if (numPositions == 4) {

                    int sumOfXChanges = ((mCenterPositions[3].x - mCenterPositions[2].x) * 3) +
                        ((mCenterPositions[2].x - mCenterPositions[1].x) * 2) +
                        ((mCenterPositions[1].x - mCenterPositions[0].x) * 1);

                    int deltaX = (int)std::round((float)sumOfXChanges / 6.0);

                    int sumOfYChanges = ((mCenterPositions[3].y - mCenterPositions[2].y) * 3) +
                        ((mCenterPositions[2].y - mCenterPositions[1].y) * 2) +
                        ((mCenterPositions[1].y - mCenterPositions[0].y) * 1);

                    int deltaY = (int)std::round((float)sumOfYChanges / 6.0);

                    mPredictedNxtPos.x = mCenterPositions.back().x + deltaX;
                    mPredictedNxtPos.y = mCenterPositions.back().y + deltaY;

                } else if (numPositions >= 5) {

                    int sumOfXChanges = ((mCenterPositions[numPositions - 1].x - mCenterPositions[numPositions - 2].x) * 4) +
                        ((mCenterPositions[numPositions - 2].x - mCenterPositions[numPositions - 3].x) * 3) +
                        ((mCenterPositions[numPositions - 3].x - mCenterPositions[numPositions - 4].x) * 2) +
                        ((mCenterPositions[numPositions - 4].x - mCenterPositions[numPositions - 5].x) * 1);

                    int deltaX = (int)std::round((float)sumOfXChanges / 10.0);

                    int sumOfYChanges = ((mCenterPositions[numPositions - 1].y - mCenterPositions[numPositions - 2].y) * 4) +
                        ((mCenterPositions[numPositions - 2].y - mCenterPositions[numPositions - 3].y) * 3) +
                        ((mCenterPositions[numPositions - 3].y - mCenterPositions[numPositions - 4].y) * 2) +
                        ((mCenterPositions[numPositions - 4].y - mCenterPositions[numPositions - 5].y) * 1);

                    int deltaY = (int)std::round((float)sumOfYChanges / 10.0);

                    mPredictedNxtPos.x = mCenterPositions.back().x + deltaX;
                    mPredictedNxtPos.y = mCenterPositions.back().y + deltaY;

                } else {
                    // should never get here
                }

            }

        }
    }
}
