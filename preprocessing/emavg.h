#ifndef _EM_AVG_
#define _EM_AVG_

class EMAvg {

public:
#ifdef EM_AVG_USE_DOUBLE
    typedef double real;
#else
    typedef float real;
#endif

    /**
     * Constructs the moving average, starting with #startValue# as its value.
     * The #alphaOrN# argument has two options:
     * - if <= 1 then it's used directly as the alpha value
     * - if > 1 then it's used as the "number of items that are considered from the past" (*) 
     * (*) Of course this is an approximation. It actually sets the alpha value to 2 / (n - 1)
     */
    EMAvg(float aphaOrN=0.1, real startValue=0);

    /**
     * Resets the moving average to #startValue# 
     */
    void reset(real startValue);

    /**
     * This function will reset the moving average with an appropriate value, waiting for convergence.
     * It is especially useful for small alpha values. Call it with a pointer to a function that returns
     * the value that you want 
     */
    void reset(real (*valueFunc)(void));

    /**
     * Updates the moving average with new value #value#
     * @param value - value to update the moving average to
     * @return the current moving average. 
     */
    real update(real value);

    /**
     * Returns the value of the moving average. 
     */
    real get() const { return _value; }

private:
    // The alpha (mixing) variable (in [0, 1]).
    float _alpha;

    // The current value of the exponential moving average.
    real _value;
};

#endif // EM_AVG_
