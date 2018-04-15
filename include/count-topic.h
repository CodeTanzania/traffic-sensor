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
                    MIDDLE_COUNT_MOVING_AWAY,
                    MIDDLE_COUNT_MOVING_TOWARDS,
                    UPPER_COUNT_MOVING_AWAY,
                    UPPER_COUNT_MOVING_TOWARDS,
                    DOWN_COUNT_MOVING_AWAY,
                    DOWN_COUNT_MOVING_TOWARDS
                };

                explicit CountTopic(): _middle_total_moving_away{0}, _middle_total_moving_towards{0} {}

                void setCount(const CountDirection &direction, const unsigned long count)
                {
                    if (direction == MIDDLE_COUNT_MOVING_AWAY)
                    {
                        _middle_total_moving_away = count;
                    }
                    else if (direction == MIDDLE_COUNT_MOVING_TOWARDS)
                    {
                        _middle_total_moving_towards = count;
                    }
                    else if (direction == UPPER_COUNT_MOVING_AWAY)
                    {
                        _upper_total_moving_away = count;
                    }
                    else if (direction == UPPER_COUNT_MOVING_TOWARDS)
                    {
                        _upper_total_moving_towards = count;
                    }
                    else if (direction == DOWN_COUNT_MOVING_AWAY)
                    {
                        _down_total_moving_away = count;
                    }
                    else if (direction == DOWN_COUNT_MOVING_TOWARDS)
                    {
                        _down_total_moving_towards = count;
                    }
                }

                unsigned long getCount(const CountDirection &direction) const
                {
                    if (direction == MIDDLE_COUNT_MOVING_AWAY)
                    {
                        return _middle_total_moving_away;
                    }
                    else if (direction == MIDDLE_COUNT_MOVING_TOWARDS)
                    {
                        return _middle_total_moving_towards;
                    }
                    else if (direction == UPPER_COUNT_MOVING_AWAY)
                    {
                        return _upper_total_moving_away;
                    }
                    else if (direction == UPPER_COUNT_MOVING_TOWARDS)
                    {
                        return _upper_total_moving_towards;
                    }
                    else if (direction == DOWN_COUNT_MOVING_AWAY)
                    {
                        return _down_total_moving_away;
                    }
                    else if (direction == DOWN_COUNT_MOVING_TOWARDS)
                    {
                        return _down_total_moving_towards;
                    }
                    else
                    {
                        return 0;
                    }
                }
            private:
                unsigned long _middle_total_moving_away;
                unsigned long _middle_total_moving_towards;
                unsigned long _upper_total_moving_away;
                unsigned long _upper_total_moving_towards;
                unsigned long _down_total_moving_away;
                unsigned long _down_total_moving_towards;
            };
        }
    }
}

#endif
