// main.cpp
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Dataset.h"
#include "ImageInfo.h"

static void print_info(const ImageInfo& info, const char* prefix = "") {
    std::cout << prefix << "ImageInfo{\n"
              << "  name: " << info.name() << "\n"
              << "  image: " << info.path() << "\n"
              << "  label: " << info.pathLabel() << "\n"
              << "}\n";
}

static void try_open_with_opencv(const std::string& img_path) {
    std::cout << "Trying to open image with OpenCV: " << img_path << "\n";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] imread failed. File not found or unreadable: " << img_path << "\n";
    } else {
        std::cout << "  Loaded OK. Size: " << img.cols << "x" << img.rows
                  << "  Channels: " << img.channels() << "\n";
    }
}

int main() {
    try {
        // Use user-specified relative paths
        const std::string image_dir = "Dataset/Images/Images/";
        const std::string annotation_dir = "Dataset/YOLO_Annotations/YOLO_Annotations/";

        std::cout << "Checking directories exist:\n";
        std::cout << "  images:      " << image_dir
                  << (std::filesystem::exists(image_dir) ? " [OK]\n" : " [MISSING]\n");
        std::cout << "  annotations: " << annotation_dir
                  << (std::filesystem::exists(annotation_dir) ? " [OK]\n" : " [MISSING]\n");

        // Basic ImageInfo checks
        std::cout << "\n--- ImageInfo basic ---\n";
        ImageInfo ii_default;
        std::cout << "default empty(): " << (ii_default.empty() ? "true" : "false") << "\n";
        ImageInfo ii_two{"SAMPLE", image_dir + std::string("AC0.jpg"), annotation_dir + std::string("AC0.txt")};
        print_info(ii_two, "two-arg: ");
        ImageInfo ii_three{"SAMPLE3", image_dir + std::string("AC1.jpg"), annotation_dir + std::string("AC1.txt")};
        print_info(ii_three, "three-arg: ");

        // Constructors
        std::cout << "\n--- Constructors ---\n";
        Dataset ds_default;
        Dataset ds_str(image_dir, annotation_dir);
        Dataset ds_path{std::filesystem::path(image_dir), std::filesystem::path(annotation_dir)};

        std::cout << "ds_str.size() = " << ds_str.size() << "\n";
        std::cout << "ds_path.size() = " << ds_path.size() << "\n";
        std::cout << "ds_default.size() = " << ds_default.size() << "\n";
        std::cout << "roots (str): images='" << ds_str.image_root().string() << "' labels='" << ds_str.annotation_root().string() << "'\n";

        // Equality (same roots + same cursor)
        std::cout << "\n--- Equality ---\n";
        ds_str.reset();
        ds_path.reset();
        std::cout << "ds_str == ds_path ? " << (ds_str == ds_path ? "true" : "false") << "\n";
        std::cout << "ds_str != ds_default ? " << (ds_str != ds_default ? "true" : "false") << "\n";

        // at() and operator[]
        if (ds_str.size() > 0) {
            std::cout << "\n--- at() / operator[] ---\n";
            ImageInfo a0 = ds_str.at(0);
            print_info(a0, "at(0): ");
            ImageInfo b0 = ds_str[0];
            print_info(b0, "op: ");

            // Try to open with OpenCV
            try_open_with_opencv(a0.path());
        } else {
            std::cout << "\nDataset appears empty. Skipping element access tests.\n";
        }

        // Out-of-range behavior
        std::cout << "\n--- Out-of-range check ---\n";
        try {
            if (ds_str.size() == 0) {
                std::cout << "Attempting at(0) on empty dataset (should throw)...\n";
                (void)ds_str.at(0);
            } else {
                std::cout << "Attempting at(size()) (should throw)...\n";
                (void)ds_str.at(ds_str.size()); // one past end
            }
            std::cerr << "[WARNING] No exception thrown where one was expected.\n";
        } catch (const std::out_of_range& e) {
            std::cout << "Caught expected std::out_of_range: " << e.what() << "\n";
        }

        // Iteration with has_next()/next()
        std::cout << "\n--- Iteration with has_next()/next() ---\n";
        ds_str.reset();
        std::size_t count = 0;
        while (ds_str.has_next()) {
            ImageInfo cur = ds_str.next();
            if (count < 3) { // print first few to avoid spamming
                print_info(cur, "next: ");
            }
            if (count < 2) { // try opening the first couple to validate paths
                try_open_with_opencv(cur.path());
            }
            ++count;
        }
        std::cout << "Iterated " << count << " entries.\n";

        // Verify next() throws at end
        std::cout << "Calling next() at end should throw...\n";
        try {
            (void)ds_str.next();
            std::cerr << "[WARNING] next() did not throw at end.\n";
        } catch (const std::out_of_range& e) {
            std::cout << "Caught expected std::out_of_range: " << e.what() << "\n";
        }

        // Test prefix/postfix ++ (advance cursor)
        std::cout << "\n--- operator++ tests ---\n";
        ds_path.reset();
        if (ds_path.size() > 2) {
            std::cout << "Initial current_index: " << ds_path.current_index() << "\n";
            ++ds_path; // prefix
            std::cout << "After ++ds_path, current_index: " << ds_path.current_index() << "\n";
            print_info(ds_path[ds_path.current_index()], "Current after prefix++: ");

            Dataset before = ds_path++; // postfix: 'before' cursor is old
            std::cout << "After ds_path++, current_index: " << ds_path.current_index() << "\n";
            if (before.current_index() < before.size()) {
                print_info(before[before.current_index()], "Postfix returned (old position): ");
            }
            if (ds_path.current_index() < ds_path.size()) {
                print_info(ds_path[ds_path.current_index()], "Current after postfix++: ");
            }
        } else {
            std::cout << "Not enough elements to demonstrate ++ operators safely.\n";
        }

        // Comparison again after cursor movements
        std::cout << "\n--- Equality after movement ---\n";
        std::cout << "ds_str == ds_path ? " << (ds_str == ds_path ? "true" : "false") << "\n";
        std::cout << "ds_str != ds_path ? " << (ds_str != ds_path ? "true" : "false") << "\n";

        std::cout << "\nAll tests completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Unhandled exception: " << e.what() << "\n";
        return 1;
    }
}
