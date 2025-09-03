# Repo Info

- Name: 2D2D Project
- Language: C++17
- Build: CMake, OpenCV, Eigen, DCMTK
- Entry: src/maindcm.cpp builds to main.exe

## Notable modules
- blob_detection.*: blob finding and filtering
- Center_detection.*: center computation
- crop_image.*: crops to ROI
- plate12icp.*, plate12indexing.*: plate indexing & ICP
- Calibration.*: calibration pipeline
- JsonCheck.*: input validation

## Runtime outputs
- build/ and Output/<Position>/* contain intermediates and results
- Writes PD/<position>_2D.txt, DD/<position>UD.dcm, icpPlatFid.txt, final_plate_points.txt

## DICOM
- Now uses DCMTK for DICOM I/O (reading, writing cropped output).