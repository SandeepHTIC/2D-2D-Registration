#ifndef CROP_IMAGE_H
#define CROP_IMAGE_H

#include <opencv2/opencv.hpp>

class CropImage {
public:
    /**
     * Crop the input image similar to MATLAB Crop_Image:
     *  - Rescale to [0,1]
     *  - Remove white strip if present
     *  - Mask low-intensity areas
     *  - Find largest region
     *  - Expand bounding box by margin
     * 
     * @param img_in Input single-channel CV_8U image
     * @param xmin Output: ROI xmin
     * @param ymin Output: ROI ymin
     * @param xmax Output: ROI xmax
     * @param ymax Output: ROI ymax
     * @return Cropped image
     */
    static cv::Mat run(const cv::Mat& img_in, int& xmin, int& ymin, int& xmax, int& ymax);
};

#endif // CROP_IMAGE_H
