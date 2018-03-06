#ifndef TRAFFIC_SENSOR_COUNTER_H
#define TRAFFIC_SENSOR_COUNTER_H

// obtained here https://stackoverflow.com/questions/1926605/how-to-count-the-number-of-objects-created-in-c

#include <iostream>

namespace com {
    namespace github {
        namespace codetanzania {

            template <typename T>
            struct Counter
            {
                explicit Counter();
                virtual ~Counter();
                static int objectsCreated;
                static int objectsAlive;
            };

            template <typename T> Counter<T>::Counter() {
                objectsCreated++;
                objectsAlive++;
            }

            template <typename T> Counter<T>::~Counter() {
                if (objectsAlive > 0)
                    --objectsAlive;
            }

            template <typename T> int Counter<T>::objectsCreated( 0 );
            template <typename T> int Counter<T>::objectsAlive( 0 );
        }
    }
}

#endif // TRAFFIC_SENSOR_COUNTER_H
