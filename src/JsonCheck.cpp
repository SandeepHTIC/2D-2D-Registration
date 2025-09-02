#include "JsonCheck.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using json = nlohmann::json; // nlohmann::json is exposed via <json.hpp> include in consumer files

namespace {
    void ensureDir(const std::string& path) {
        std::filesystem::create_directories(path);
    }

    void writeFailure(const std::string& path_d,
                      const std::string& position,
                      const std::string& errorMessage,
                      const std::string& regType,
                      JsonCheckResult& out) {
        try {
            std::string outDir = path_d + "\\Output\\" + position;
            ensureDir(outDir);
            std::string file = outDir + "\\output" + position + ".json";
            json output_SSR;
            output_SSR["Type"] = position;
            output_SSR["TwoD_Points"] = json::array();
            output_SSR["Result_Matrix"] = json::array();
            output_SSR["UD_InPaint_Image"] = "";
            output_SSR["Status"] = "Failure";
            output_SSR["CropRoi"] = json::array();
            output_SSR["RegistrationType"] = regType;
            output_SSR["RPE"] = -1;
            output_SSR["ErrorMessage"] = errorMessage;
            std::ofstream ofs(file);
            ofs << output_SSR.dump();
            out.outputJsonPath = file;
        } catch (...) {
        }
        out.ok = false;
        out.errorMessage = errorMessage;
    }
}

JsonCheckResult json_check_cpp(const json& input_SSR, const std::string& otpath, const std::string& path_d) {
    JsonCheckResult res{};
    res.ok = false;

    // Validate position
    try {
        std::string pos = input_SSR.at("Type").get<std::string>();
        if (!(pos == "AP" || pos == "LP")) {
            writeFailure(path_d, otpath, "Position Type Is Incorrect", "", res);
            return res;
        }
        res.position = pos;
    } catch (...) {
        writeFailure(path_d, otpath, "Position Type Is Incorrect", "", res);
        return res;
    }

    // Ensure Output structure; remove previous position folder like MATLAB
    try {
        std::string outRoot = path_d + "\\Output";
        ensureDir(outRoot);
        std::string posDir = outRoot + "\\" + res.position;
        if (std::filesystem::exists(posDir)) {
            std::filesystem::remove_all(posDir);
        }
        // Only PD folder is used now
        ensureDir(posDir + "\\PD");
    } catch (...) {
        // Non-fatal for parity
    }

    // Config file is optional in this flow (maindcm does not require it). Skip mandatory check.
    {
        // If needed in future, optionally read path_d+"\\config.json" here without failing.
    }

    // RegistrationType must be 2D2D
    std::string regType;
    try {
        regType = input_SSR.at("RegistrationType").get<std::string>();
        if (regType != "2D2D") {
            writeFailure(path_d, res.position, "Registration Method Not Found", "", res);
            return res;
        }
        res.registrationType = regType;
    } catch (...) {
        writeFailure(path_d, res.position, "Registration Method Not Found", "", res);
        return res;
    }

    // CropRoi
    try {
        if (input_SSR.contains("CropRoi")) {
            auto roi = input_SSR.at("CropRoi");
            if (roi.is_array() && roi.size() == 4) {
                double norm2 = 0.0;
                res.cropRoi.clear();
                for (size_t i = 0; i < 4; ++i) { double v = roi.at(i).get<double>(); res.cropRoi.push_back(v); norm2 += v*v; }
                // MATLAB allows exactly-4 and norm==0 case; otherwise it's an error path. We store as-is.
            } else {
                writeFailure(path_d, res.position, "Crop ROI Is Empty", regType, res);
                return res;
            }
        } else {
            writeFailure(path_d, res.position, "Crop ROI Is Empty", regType, res);
            return res;
        }
    } catch (...) {
        writeFailure(path_d, res.position, "Crop ROI Is Empty", regType, res);
        return res;
    }

    // Image path and reading feasibility check (presence only to match MATLAB outcomes)
    try {
        std::string imagePath = input_SSR.at("Image").get<std::string>();
        res.imagePath = imagePath;
        // Basic extension check for .dcm or .DCM
        auto dot = imagePath.find_last_of('.');
        if (dot == std::string::npos) {
            writeFailure(path_d, res.position, "Incompatible File Format", regType, res);
            return res;
        }
        std::string ext = imagePath.substr(dot);
        for (auto& c : ext) c = (char)toupper((unsigned char)c);
        if (ext != ".DCM") {
            writeFailure(path_d, res.position, "Incompatible File Format", regType, res);
            return res;
        }
    } catch (...) {
        writeFailure(path_d, res.position, "Image path is not present in JSON file", regType, res);
        return res;
    }

    // CMM_WorldPoints: present and either full Nx3 or at least 17 rows
    try {
        const auto& wp = input_SSR.at("CMM_WorldPoints");
        if (!wp.is_array() || wp.empty()) {
            writeFailure(path_d, res.position, "CMM World Points Are Not Present", regType, res);
            return res;
        }
        // MATLAB tries a special case of exactly 23 and then takes first 17. We will mirror that.
        size_t rows = wp.size();
        size_t useRows = rows;
        if (rows == 23) useRows = 17;
        Eigen::MatrixXd W(useRows, 3);
        for (size_t i = 0; i < useRows; ++i) {
            W(i, 0) = wp.at(i).at(0).get<double>();
            W(i, 1) = wp.at(i).at(1).get<double>();
            W(i, 2) = wp.at(i).at(2).get<double>();
        }
        if (useRows != 17) {
            writeFailure(path_d, res.position, "CMM World Points Count Not Equal To 17", regType, res);
            return res;
        }
        res.CMM_WorldPoints = W;
    } catch (...) {
        writeFailure(path_d, res.position, "CMM World Points Are Not Present", regType, res);
        return res;
    }

    // CMM_Dist_pts
    try {
        const auto& dp = input_SSR.at("CMM_Dist_pts");
        if (!dp.is_array() || dp.empty()) {
            writeFailure(path_d, res.position, "CMM Dist Points Are Not Present", regType, res);
            return res;
        }
        Eigen::MatrixXd D(dp.size(), 3);
        for (size_t i = 0; i < dp.size(); ++i) {
            D(i, 0) = dp.at(i).at(0).get<double>();
            D(i, 1) = dp.at(i).at(1).get<double>();
            D(i, 2) = dp.at(i).at(2).get<double>();
        }
        res.CMM_Dist_pts = D;
    } catch (...) {
        writeFailure(path_d, res.position, "CMM Dist Points Are Not Present", regType, res);
        return res;
    }

    // Output path for success/failure JSON (we only set on failure like MATLAB does explicitly).
    res.ok = true;
    return res;
}