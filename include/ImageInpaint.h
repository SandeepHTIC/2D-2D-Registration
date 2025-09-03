#pragma once

#include <string>
#include <opencv2/opencv.hpp>

struct InpaintResult
{
    bool ok = false;
    std::string ipPath;
    std::string flipPath;
    std::string error;
};

// Compute-only inpaint. Returns inpainted and flipped-padded images via reference.
// Does not perform any file I/O.
bool ComputeInpaintImages(const std::string &baseDir,
                          const std::string &position,
                          cv::Mat &outInpainted,
                          cv::Mat &outFlippedPadded,
                          std::string &outError);

// Legacy convenience: runs compute and writes IP.dcm and Flip.dcm under Output/<position>/IP/
InpaintResult ImageInpaint(const std::string &baseDir, const std::string &position);