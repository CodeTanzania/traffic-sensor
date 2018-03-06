#ifndef TRAFFIC_SENSOR_ARROW_HEAD_H
#define TRAFFIC_SENSOR_ARROW_HEAD_H

#include <cv.h>
#include <math.h>

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            class ArrowHead {
            public:

                enum ArrowStyle {
                    ARROW_OPEN,
                    ARROW_SOLID,
                    ARROW_FILLED,
                    ARROW_DIAMOND,
                    ARROW_DIAMOND_FILLED,
                    ARROW_CIRCLE,
                    ARROW_CIRCLE_FILLED
                };

                explicit ArrowHead(const cv::Point &head, const cv::Point &tail, const int lineThickness = 1, const int lineType = 8, const double len = 60.0, const double angle = M_PI_4);

                void setHead(const cv::Point &head);
                cv::Point head() const;

                void setTail(const cv::Point &end);
                cv::Point tail() const;

                void setLength(const double length);
                double length() const;

                void setOrientation(const double angle);
                double orientation() const;

                void setLineThickness(const int thickness);
                int lineThickness() const;

                void setLineType(const int type);
                int lineType() const;

                virtual void draw(cv::Mat &frame) = 0;
            protected:
                // member variables
                double mLength;
                double mOrientation;
                int mLineThickness;
                int mLineType;
                cv::Point mHead;
                cv::Point mTail;
                // member functions
                void computeVertices(cv::Point &leftVertex, cv::Point &rightVertex);
            };
        }
    }
}

#endif // TRAFFIC_SENSOR_ARROW_HEAD_H
