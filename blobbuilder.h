#ifndef BLOBREFS_H
#define BLOBREFS_H

#include <cv.hpp>
#include <stddef.h>
#include <vector>

#include "blob.h"
#include "blobtracker.h"

namespace com {
    namespace github {
        namespace codetanzania {

            // class BlobTracker;

            class BlobBuilder {
            public:
                static Blob *create(const std::vector<cv::Point> &cnt, const BlobTracker &tracker);
            private:
                BlobBuilder() {}
            };

            Blob *BlobBuilder::create(const std::vector<cv::Point> &cnt, const BlobTracker &tracker) {
                Blob *blob = new Blob(cnt);
                if (tracker.paramsFilter(*blob)) {
                    return blob;
                }
                delete blob;
                return nullptr;
            }
        }
    }
}

#endif // BLOBREFS_H
