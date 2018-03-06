#ifndef SUBJECT_H
#define SUBJECT_H

#include <memory>
#include <vector>

#include "observer.h"


namespace com
{
    namespace github
    {
        namespace kmoz
        {
            template<class T>
            class Subject
            {

            public:
                explicit Subject() {}
                // destructor
                ~Subject() = default;

                void attach(Observer<T> *observable);
                void detach(Observer<T> *observable);
                void notify(const T &topic);
            private:
                std::vector<Observer<T> *> _observers;
            };

            template <class T>
            void Subject<T>::notify(const T &topic)
            {
                for (const auto observer : _observers)
                {
                    observer->update(topic);
                }
            }

            template <class T>
            void Subject<T>::attach(Observer<T> *observer)
            {
                _observers.push_back(observer);
            }

            template <class T>
            void Subject<T>::detach(Observer<T> *observer)
            {
                unsigned index = -1, k;
                for (k = 0; k < _observers.size(); k++)
                {
                    if (_observers[k] == observer)
                    {
                        index = k;
                        break;
                    }
                }

                if (index != -1)
                {
                    _observers.erase(_observers.begin(), index);
                }
            }
        }
    }
}

#endif // SUBJECT_H
