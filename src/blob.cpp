#include "blob.h"
#include "trackutils.h"

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
                predictBlobNextStep(*this);
            }

        }
    }
}
