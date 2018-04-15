
#include <algorithm>
#include <cmath>
#include "emavg.h"

using namespace std;

EMAvg::EMAvg(float alphaOrN, real startValue) : _value(startValue) {
    alphaOrN = max(alphaOrN, 0.0f); // make sure alphaOrN >= 0
    _alpha = (alphaOrN > 1 ?
        2 / (alphaOrN + 1) :
        alphaOrN);
}

void EMAvg::reset(real startValue) {
    _value = startValue;
}

void EMAvg::reset(real (*valueFunc)(void)) {
    // Source: http://www.had2know.com/finance/exponential-moving-average-ema-calculator.html
    // a = 2 / (n + 1) ==> n => 2/a - 1 ==> (n - 1) / 2 = 1/a - 1
    int n = ceil(1.0f / _alpha - 1);
    real avg = 0.0;
    for (int i = 0; i < n; i++)
        avg += valueFunc();
    avg /= n;

    reset(avg);
}

EMAvg::real EMAvg::update(real value) {
    return (_value -= _alpha * (_value - value));
}
