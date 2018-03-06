#ifndef TRAFFIC_SENSOR_COUNT_TOPIC
#define TRAFFIC_SENSOR_COUNT_TOPIC

#include "patterns/topic.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            class CountTopic : public com::github::kmoz::Topic
            {
            public:
                enum CountDirection
                {
                    COUNT_MOVING_AWAY,
                    COUNT_MOVING_TOWARDS
                };

                explicit CountTopic(): _total_moving_away{0}, _total_moving_towards{0} {}

                void setCount(const CountDirection &direction, const unsigned long count)
                {
                    if (direction == COUNT_MOVING_AWAY)
                    {
                        _total_moving_away = count;
                    }
                    else
                    {
                        _total_moving_towards = count;
                    }
                }

                unsigned long getCount(const CountDirection &direction) const
                {
                    if (direction == COUNT_MOVING_AWAY)
                    {
                        return _total_moving_away;
                    }
                    else
                    {
                        return _total_moving_towards;
                    }
                }
            private:
                unsigned long _total_moving_away;
                unsigned long _total_moving_towards;
            };
        }
    }
}

#endif
