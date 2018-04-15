#ifndef _PREPROCESSING_WMAVG_H_
#define _PREPROCESSING_WMAVG_H_

/**
 * Compute the moving average by multipling weights to the values and divide
 * by the triangular sum of the weights.
 * see https://www.geeksforgeeks.org/sum-series-1-3-6-10-triangular-numbers/ on how
 * to formulate triangular sum.
 */
class WMAvg {
public:

    #ifdef WMAVG_USE_DOUBLE
        typedef double real;
    #else
        typedef float real;
    #endif

    template<class InputIterator>
    static real get(InputIterator first, InputIterator last) {
        if (first == last) {
            return *first;
        }
        unsigned n = 0;
        real sum = 0.0;
        real avg = 0.0;

        while(first != last) {
            sum += n * (n + 1) / 2;
            avg += (*first++) * (++n);
        } // end while

        avg /= ((sum == 0.0) ? 1.0 : sum);
        return avg; 
    }
};

#endif // _PREPROCESSING_WMAVG_H_