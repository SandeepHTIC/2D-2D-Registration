#pragma once

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcpixel.h>

#include <opencv2/opencv.hpp>
#include <string>

inline bool writeDICOM_DCMTK(const cv::Mat &img, const std::string &outPath)
{
    if (img.empty())
        return false;
    cv::Mat mono;
    if (img.channels() == 1)
        mono = img;
    else
        cv::cvtColor(img, mono, cv::COLOR_BGR2GRAY);
    if (!mono.isContinuous())
        mono = mono.clone();

    DcmFileFormat file;
    DcmDataset *ds = file.getDataset();

    // Generate UIDs
    char sopInstanceUID[100];
    dcmGenerateUniqueIdentifier(sopInstanceUID);
    char seriesInstanceUID[100];
    dcmGenerateUniqueIdentifier(seriesInstanceUID);
    char studyInstanceUID[100];
    dcmGenerateUniqueIdentifier(studyInstanceUID);

    ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    ds->putAndInsertString(DCM_SOPInstanceUID, sopInstanceUID);
    ds->putAndInsertString(DCM_StudyInstanceUID, studyInstanceUID);
    ds->putAndInsertString(DCM_SeriesInstanceUID, seriesInstanceUID);

    ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows, static_cast<Uint16>(mono.rows));
    ds->putAndInsertUint16(DCM_Columns, static_cast<Uint16>(mono.cols));
    ds->putAndInsertUint16(DCM_BitsAllocated, 8);
    ds->putAndInsertUint16(DCM_BitsStored, 8);
    ds->putAndInsertUint16(DCM_HighBit, 7);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0);

    const Uint8 *pixelPtr = mono.ptr<Uint8>(0);
    const unsigned long numBytes = static_cast<unsigned long>(mono.total());
    OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData, pixelPtr, numBytes);
    if (!st.good())
        return false;

    st = file.saveFile(outPath.c_str(), EXS_LittleEndianExplicit);
    return st.good();
}