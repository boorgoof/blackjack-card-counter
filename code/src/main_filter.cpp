// main.cpp
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include "../include/ImageFilter.h"


int main() {
    const std::string path =
        "/mnt/d/LAB_computer_vision/real_final_project/blackjack-card-counter/data/datasets/single_cards/Images/Images/2C0.jpg";

    // carica
    cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);

    // ridimensiona a 0.5x prima dei filtri
    cv::Mat src_half;
    cv::resize(src, src_half, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    // pipeline semplice
    ImageFilter pipeline;
    pipeline.add_filter("clahe",   Filters::CLAHE_contrast_equalization, 2, 8);
    pipeline.add_filter("gauss",   Filters::gaussian_blur, cv::Size(5,5));
    pipeline.add_filter("unsharp", Filters::unsharp_mask, 1.0, 0.25);

    // applica filtri sull'immagine ridotta
    cv::Mat dst = pipeline.apply_filters(src_half);

    // affianca (sinistra: originale ridotta; destra: filtrata)
    cv::Mat side(src_half.rows, src_half.cols * 2, CV_8UC3);
    src_half.copyTo(side(cv::Rect(0, 0, src_half.cols, src_half.rows)));
    dst.copyTo(side(cv::Rect(src_half.cols, 0, src_half.cols, src_half.rows)));

    // etichette
    cv::putText(side, "Questa e l'immagine (0.5x)",
                {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 4, cv::LINE_AA);
    cv::putText(side, "Questa e l'immagine (0.5x)",
                {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255,255,255}, 2, cv::LINE_AA);

    cv::putText(side, "Questa e l'immagine dopo il filtro",
                {src_half.cols + 20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 4, cv::LINE_AA);
    cv::putText(side, "Questa e l'immagine dopo il filtro",
                {src_half.cols + 20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255,255,255}, 2, cv::LINE_AA);

    // mostra
    cv::namedWindow("show0", cv::WINDOW_NORMAL);
    cv::imshow("show0", side);
    cv::waitKey(0);
    return 0;
}