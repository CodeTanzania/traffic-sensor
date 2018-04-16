#include <numeric>

#include "blob.h"
#include "preprocessing/wmavg.h"

using namespace cv;
using namespace std;

namespace com {
    namespace github {
        namespace codetanzania {

            static void predictBlobNextStep(Blob &blob) {
                vector<Point> centers = blob.centers();
                unsigned N = centers.size();

                int xvals[N];
                int yvals[N];
                int xDeltas[N];
                int yDeltas[N];

                for (unsigned i = 0; i < N; i++) {
                    xvals[i] = centers[i].x;
                    yvals[i] = centers[i].y;
                }

                adjacent_difference(xvals, xvals + N, xDeltas);
                adjacent_difference(yvals, yvals + N, yDeltas);

                int avgXDelta = 0, avgYDelta = 0;
                float sum = 0.0f;
                for (unsigned i = 0; i < N; i++) {
                    avgXDelta += (xDeltas[i] * (i));
                    avgYDelta += (yDeltas[i] * (i));
                    sum += i * (i + 1) / 2; // sum of triangular numbers
                }

                sum = (sum == 0.0f) ? 1.0f : sum; // avoid divide by 0 errors

                avgXDelta /= sum;
                avgYDelta /= sum;

                Point prediction(centers.back().x + static_cast<int>(avgXDelta),
                    centers.back().y + static_cast<int>(avgYDelta));

                blob.setNextPosition(prediction);
            }

        }
    }
}