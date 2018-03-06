#ifndef TRAFFIC_SENSOR_LOGGING_OBSERVER_H
#define TRAFFIC_SENSOR_LOGGING_OBSERVER_H

#include <algorithm>
#include <chrono>
#include <ctime>

#include "patterns/observer.h"
#include "count-topic.h"

namespace com
{
    namespace github
    {
        namespace codetanzania
        {
            class LoggingObserver : public com::github::kmoz::Observer<CountTopic>
            {
            public:
                explicit LoggingObserver() {}
                void update(const CountTopic &topic) {
                    auto now = std::chrono::system_clock::now();
                    std::time_t log_time = std::chrono::system_clock::to_time_t(now);
                    std::string timestr = std::ctime(&log_time);
                    timestr.erase(std::remove(timestr.begin(), timestr.end(), '\n'), timestr.end());
                    std::cout << "[LOG "
                              << timestr
                              << "] Moving Away: "
                              << topic.getCount(CountTopic::CountDirection::COUNT_MOVING_AWAY)
                              << " Moving Toward: "
                              << topic.getCount(CountTopic::CountDirection::COUNT_MOVING_TOWARDS)
                              << std::endl;
                }
            };
        }
    }
}

#endif
