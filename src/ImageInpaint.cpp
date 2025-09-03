#include "ImageInpaint.h"
#include "readdicom.h"
#include "writedicom.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>

using namespace cv;

static std::vector<cv::Point2f> loadXY(const std::string &p, int mode)
{
    std::vector<cv::Point2f> pts;
    std::ifstream f(p);
    if (!f)
        return pts; // return empty if missing
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v)
            vals.push_back(v);
        if (vals.size() >= 2)
        {
            if (mode == 0)
            { // PD _2D.txt → first two columns x y
                pts.emplace_back(static_cast<float>(vals[0]), static_cast<float>(vals[1]));
            }
            else if (mode == 1)
            { // UD_pts.txt → x y
                pts.emplace_back(static_cast<float>(vals[0]), static_cast<float>(vals[1]));
            }
            else
            { // icp → if >2 cols, take 2nd & 3rd (x,y), else first two
                if (vals.size() > 2)
                    pts.emplace_back(static_cast<float>(vals[1]), static_cast<float>(vals[2]));
                else
                    pts.emplace_back(static_cast<float>(vals[0]), static_cast<float>(vals[1]));
            }
        }
    }
    return pts;
}

bool ComputeInpaintImages(const std::string &baseDir,
                          const std::string &position,
                          cv::Mat &outInpainted,
                          cv::Mat &outFlippedPadded,
                          std::string &outError)
{
    try
    {
        // Build paths
        std::string base = baseDir + "\\Output\\" + position + "\\";
        std::string dd = base + "DD\\";
        std::string pd = base + "PD\\";

        std::string udPath = dd + position + "UD.dcm";
        std::string blobPath = pd + position + "bw.png";
        std::string ptsPath = pd + position + "_2D.txt";
        std::string icpPath = dd + position + "_icp_2D.txt";
        std::string ptsPPath = dd + position + "UD_pts.txt";

        // Read inputs
        cv::Mat image = readDICOM_DCMTK(udPath); // 8-bit mono
        if (image.empty())
            throw std::runtime_error("Failed to read DICOM: " + udPath);
        if (image.channels() != 1)
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

        cv::Mat blobImg = cv::imread(blobPath, cv::IMREAD_GRAYSCALE);
        if (blobImg.empty())
            throw std::runtime_error("Failed to read blob image: " + blobPath);
        // Ensure binary mask (0/255)
        cv::threshold(blobImg, blobImg, 0, 255, cv::THRESH_BINARY);

        std::vector<cv::Point2f> pts = loadXY(ptsPath, 0);
        std::vector<cv::Point2f> ptsP = loadXY(ptsPPath, 1);
        std::vector<cv::Point2f> icpPts = loadXY(icpPath, 2);
        pts.insert(pts.end(), ptsP.begin(), ptsP.end());
        pts.insert(pts.end(), icpPts.begin(), icpPts.end());

        // Connected components on blob image
        cv::Mat labels, stats, centroids;
        int nlabels = cv::connectedComponentsWithStats(blobImg, labels, stats, centroids, 8, CV_32S);
        (void)nlabels;

        // Select components that contain any of the points
        std::set<int> selectedLabels;
        for (const auto &p : pts)
        {
            int x = static_cast<int>(std::round(p.x));
            int y = static_cast<int>(std::round(p.y));
            if (x >= 0 && x < labels.cols && y >= 0 && y < labels.rows)
            {
                int lbl = labels.at<int>(y, x);
                if (lbl > 0)
                    selectedLabels.insert(lbl); // skip background label 0
            }
        }

        cv::Mat mask = cv::Mat::zeros(blobImg.size(), CV_8U);
        for (int lbl : selectedLabels)
        {
            cv::Mat sel = (labels == lbl);
            mask.setTo(255, sel);
        }

        // Dilate with disk radius 3 (7x7 ellipse)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::Mat maskDil;
        cv::dilate(mask, maskDil, kernel);

        // Inpaint (Telea)
        cv::inpaint(image, maskDil, outInpainted, 3.0, cv::INPAINT_TELEA);

        // Flip vertically and pad to [970, 1052] on bottom/right
        cv::Mat flipped;
        cv::flip(outInpainted, flipped, 0); // vertical flip
        const int targetRows = 970, targetCols = 1052;
        int addRows = std::max(0, targetRows - flipped.rows);
        int addCols = std::max(0, targetCols - flipped.cols);
        cv::copyMakeBorder(flipped, outFlippedPadded, 0, addRows, 0, addCols, cv::BORDER_CONSTANT, cv::Scalar(0));
        return true;
    }
    catch (const std::exception &e)
    {
        outError = e.what();
        return false;
    }
}

InpaintResult ImageInpaint(const std::string &baseDir, const std::string &position)
{
    InpaintResult res;
    cv::Mat inpainted, flippedPadded;
    std::string err;
    std::string base = baseDir + "\\Output\\" + position + "\\";
    std::string ipd = base + "IP\\";
    std::filesystem::create_directories(ipd);
    std::string outIP = ipd + position + "IP.dcm";
    std::string outFlip = ipd + position + "Flip.dcm";

    if (!ComputeInpaintImages(baseDir, position, inpainted, flippedPadded, err))
    {
        res.ok = false;
        res.error = err;
        return res;
    }
    if (!writeDICOM_DCMTK(inpainted, outIP))
    {
        res.ok = false;
        res.error = std::string("Failed to write IP DICOM: ") + outIP;
        return res;
    }
    if (!writeDICOM_DCMTK(flippedPadded, outFlip))
    {
        res.ok = false;
        res.error = std::string("Failed to write Flip DICOM: ") + outFlip;
        return res;
    }
    res.ok = true;
    res.ipPath = outIP;
    res.flipPath = outFlip;
    return res;
}