#ifndef _PATTERNS_OBSERVER_H_
#define _PATTERNS_OBSERVER_H_

namespace com
{
    namespace github
    {
        namespace kmoz
        {
            template <class T>
            class Observer {
            public:
                virtual void update(const T &topic) = 0;
            };
        }
    }
}

#endif // _PATTERNS_OBSERVER_H_
