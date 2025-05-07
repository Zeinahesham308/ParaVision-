#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace std;
using namespace cv;

// Generates a high-pass filter kernel of given size
vector<vector<int>> generateHighPassKernel(int size) {
    int centerValue = size * size - 1;
    vector<vector<int>> kernel(size, vector<int>(size, -1));
    kernel[size / 2][size / 2] = centerValue;
    return kernel;
}

// Applies the kernel manually to a pixel with replicate padding
int applyKernel(const Mat& image, const vector<vector<int>>& kernel, int x, int y) {
    int kSize = kernel.size();
    int half = kSize / 2;
    int sum = 0;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            int xi = min(max(x + i, 0), image.rows - 1); // replicate padding
            int yj = min(max(y + j, 0), image.cols - 1);
            int pixel = static_cast<int>(image.at<uchar>(xi, yj));
            int weight = kernel[i + half][j + half];
            sum += pixel * weight;
        }
    }

    // Clamp result between 0 and 255
    return min(max(sum, 0), 255);
}

int main(int argc, char* argv[]) {
    string inputPath = "./inputs/input.jpg";
    string outputPath = "./outputs/HPF_SEQ_outputRes.png";
    string timePath = "./outputs/time.txt";

    // Load image in color
    Mat color = imread(inputPath, IMREAD_COLOR);
    if (color.empty()) {
        cerr << "Error: Cannot load image at " << inputPath << endl;
        return 1;
    }



    // Generate a 3×3 default high-pass kernel
    int kernelSize = 3;
    if (argc > 1) {
      
        kernelSize = atoi(argv[1]);
    }
    vector<vector<int>> kernel = generateHighPassKernel(kernelSize);

    // Measure time
    auto start = chrono::high_resolution_clock::now();

    //  convert to grayscale
    Mat input(color.rows, color.cols, CV_8UC1);
    for (int i = 0; i < color.rows; ++i) {
        for (int j = 0; j < color.cols; ++j) {
            Vec3b pixel = color.at<Vec3b>(i, j); // BGR

            uchar gray = static_cast<uchar>((pixel[0] + pixel[1] + pixel[2]) / 3);
            // uchar gray = static_cast<uchar>(0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2]);
            input.at<uchar>(i, j) = gray;
        }
    }

    // Create output image
    Mat output = Mat::zeros(input.size(), CV_8UC1);


    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output.at<uchar>(i, j) = applyKernel(input, kernel, i, j);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;

    // Save result image
    imwrite(outputPath, output);
    cout << "High-pass filtered image saved to: " << outputPath << endl;

    // Save execution time
    ofstream timeFile(timePath);
    if (timeFile.is_open()) {
        timeFile<<"Execution time: " << duration.count()<<endl;
        timeFile << "Kernal size used : " << kernelSize << endl;
        timeFile.close();
        cout << "Execution time: " << duration.count() << " ms (saved to time.txt)" << endl;
    }
    else {
        cerr << "Could not write execution time to file." << endl;
    }

    return 0;
}
