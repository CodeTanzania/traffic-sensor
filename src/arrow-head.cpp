
#include "arrow-head.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            // const int lineThickness = 1, const int lineType = 8,
            // constructor
            ArrowHead::ArrowHead(const cv::Point &head, const cv::Point &tail, const int lineThickness, const int lineType, const double len, const double orientation)
            {
                this->mLength = len;
                this->mOrientation = orientation;
                this->mHead = head;
                this->mTail = tail;
            } // end constructor

            // compute vertices of the arrow
            void ArrowHead::computeVertices(cv::Point &leftVertex, cv::Point &rightVertex)
            {
                double angle = std::atan2(mTail.y - mHead.y, mTail.x - mHead.x) + M_PI;
                leftVertex.x = mTail.x + mLength * std::cos(angle - mOrientation);
                leftVertex.y = mTail.y + mLength * std::sin(angle - mOrientation);
                rightVertex.x = mTail.x + mLength * std::cos(angle + mOrientation);
                rightVertex.y = mTail.y + mLength * std::sin(angle + mOrientation);
            } // end computeVertices

            // starting point, joining the two vertices
            void ArrowHead::setHead(const cv::Point &head)
            {
                mHead = head;
            } // end start

            // get starting point, joining the two vertices
            cv::Point ArrowHead::head() const
            {
                return mHead;
            } // end get starting point

            // set tail or the arrow
            void ArrowHead::setTail(const cv::Point &tail)
            {
                mTail = tail;
            } // end tail

            // get tail of the arrow
            cv::Point ArrowHead::tail() const
            {
                return mTail;
            } // end arrow tail

            // set arrow orientation w.r.t vertical axis
            void ArrowHead::setOrientation(const double angle)
            {
                mOrientation = angle;
            } // end arrow orientation

            // get arrow orientation w.r.t vertical axis
            double ArrowHead::orientation() const
            {
                return mOrientation;
            } // end arrow orientation

            // set arrow length
            void ArrowHead::setLength(const double length)
            {
                mLength = length;
            } // end set arrow length

            // get arrow length
            double ArrowHead::length() const
            {
                return mLength;
            } // end arrow length

            // set line thickness
            void ArrowHead::setLineThickness(const int thickness)
            {
                mLineThickness = thickness;
            } // end get line thickness

            // get line thickness
            int ArrowHead::lineThickness() const
            {
                return mLineThickness;
            } // end get line thickness

            // set line type
            void ArrowHead::setLineType(const int type) {
                mLineType = type;
            } // end set line type

            // get line type
            int ArrowHead::lineType() const {
                return mLineType;
            } // end get line type

        }
    }
}
