#ifndef TRAFFIC_SENSOR_ARROW_OPEN_H
#define TRAFFIC_SENSOR_ARROW_OPEN_H

#include <opencv2/opencv.hpp>

#include "arrow-head.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            class ArrowOpen : ArrowHead
            {
            public:
                ArrowOpen(const cv::Point &head,const cv::Point &tail, const cv::Scalar &color = cv::Scalar(0, 0, 0), const int lineThickness = 1, const int lineType = 8, const double len = 60.0, const double angle = M_PI_4) :
                    ArrowHead(head, tail, lineThickness, lineType, len, angle)
                {

                }

                void setColor(const cv::Scalar &color)
                {
                    mColor = color;
                }

                cv::Scalar color() const
                {
                    return mColor;
                }

                void draw(cv::Mat &frame) {
                    cv::Point leftVertex;
                    cv::Point rightVertex;

                    cv::line(frame, head(), leftVertex, color(), lineThickness(), lineType());
                    cv::line(frame, head(), rightVertex, color(), lineThickness(), lineType());
                }

            private:
                cv::Scalar mColor;
            };
        }
    }
}

#endif // TRAFFIC_ARROW_OPEN_H
