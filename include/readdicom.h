#pragma once

// DCMTK must have config first
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcpixel.h>
#include <dcmtk/dcmjpeg/djdecode.h> // JPEG decoders
#include <dcmtk/dcmdata/dcrledrg.h> // RLE decoder registration

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

inline cv::Mat readDICOM_DCMTK(const std::string &filepath)
{
    static bool codecs_registered = false;
    if (!codecs_registered)
    {
        DJDecoderRegistration::registerCodecs();
        DcmRLEDecoderRegistration::registerCodecs();
        codecs_registered = true;
    }

    DcmFileFormat file;
    OFCondition cond = file.loadFile(filepath.c_str());
    if (!cond.good())
        throw std::runtime_error("DCMTK loadFile failed: " + std::string(cond.text()));

    DcmDataset *ds = file.getDataset();

    // Basic attributes
    Uint16 rows = 0, cols = 0, samplesPerPixel = 1, bitsAllocated = 8;
    ds->findAndGetUint16(DCM_Rows, rows);
    ds->findAndGetUint16(DCM_Columns, cols);
    ds->findAndGetUint16(DCM_SamplesPerPixel, samplesPerPixel);
    ds->findAndGetUint16(DCM_BitsAllocated, bitsAllocated);

    OFString photo;
    ds->findAndGetOFString(DCM_PhotometricInterpretation, photo);

    // Access pixel data for uncompressed images. For compressed cases, throw and let caller fall back to DicomImage.
    DcmElement *elem = nullptr;
    OFCondition ec = ds->findAndGetElement(DCM_PixelData, elem);
    if (!ec.good() || !elem)
        throw std::runtime_error("Missing PixelData");

    cv::Mat out;
    std::string p(photo.c_str());

    if (bitsAllocated <= 8)
    {
        Uint8 *data8 = nullptr;
        ec = elem->getUint8Array(data8);
        if (!ec.good() || !data8)
        {
            throw std::runtime_error("Compressed/unsupported 8-bit pixel data (use DicomImage fallback)");
        }
        // Create cv::Mat and clone to own memory
        int type = (samplesPerPixel == 1 ? CV_8UC1 : CV_8UC3);
        out = cv::Mat(rows, cols, type, static_cast<void *>(data8)).clone();

        // Photometric conversions if needed
        if (samplesPerPixel == 3 && (p == "YBR_FULL" || p == "YBR_FULL_422"))
        {
            cv::cvtColor(out, out, cv::COLOR_YCrCb2RGB);
        }
        if (samplesPerPixel == 3)
        {
            cv::Mat gray;
            cv::cvtColor(out, gray, cv::COLOR_RGB2GRAY);
            out = gray;
        }
    }
    else
    { // 16-bit
        Uint16 *data16 = nullptr;
        ec = elem->getUint16Array(data16);
        if (!ec.good() || !data16)
        {
            throw std::runtime_error("Compressed/unsupported 16-bit pixel data (use DicomImage fallback)");
        }
        int type16 = (samplesPerPixel == 1 ? CV_16UC1 : CV_16UC3);
        cv::Mat tmp16(rows, cols, type16, static_cast<void *>(data16));

        // Normalize to 8-bit
        if (samplesPerPixel == 1)
        {
            double minV, maxV;
            cv::minMaxLoc(tmp16, &minV, &maxV);
            tmp16.convertTo(out, CV_8U, 255.0 / (maxV - minV + 1e-9), -minV * 255.0 / (maxV - minV + 1e-9));
        }
        else
        {
            tmp16.convertTo(out, CV_8UC3, 1.0 / 256.0);
            if (p == "YBR_FULL" || p == "YBR_FULL_422")
            {
                cv::cvtColor(out, out, cv::COLOR_YCrCb2RGB);
            }
            cv::Mat gray;
            cv::cvtColor(out, gray, cv::COLOR_RGB2GRAY);
            out = gray;
        }
    }

    return out;
}