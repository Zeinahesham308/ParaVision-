#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

// Generate high-pass filter kernel
vector<vector<int>> generateHighPassKernel(int size) {
    int centerValue = size * size - 1;
    vector<vector<int>> kernel(size, vector<int>(size, -1));
    kernel[size / 2][size / 2] = centerValue;
    return kernel;
}

// Apply kernel on a pixel
int applyKernel(const Mat& image, const vector<vector<int>>& kernel, int x, int y) {
    int kSize = kernel.size();
    int half = kSize / 2;
    int sum = 0;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            int xi = min(max(x + i, 0), image.rows - 1);
            int yj = min(max(y + j, 0), image.cols - 1);
            int pixel = static_cast<int>(image.at<uchar>(xi, yj));
            int weight = kernel[i + half][j + half];
            sum += pixel * weight;
        }
    }

    return min(max(sum, 0), 255);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int kernelSize = 3;
    if (argc > 1) {

        kernelSize = atoi(argv[1]);
    }
    vector<vector<int>> kernel;
    Mat fullImage, grayImage;
    int rows, cols;


    if (rank == 0) {
        // Load and convert to grayscale
        fullImage = imread("./inputs/input.jpg", IMREAD_COLOR);

        if (fullImage.empty()) {
            cerr << "Error loading image\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        //cvtColor(fullImage, grayImage, COLOR_BGR2GRAY);
        grayImage = Mat(fullImage.rows, fullImage.cols, CV_8UC1);
        for (int i = 0; i < fullImage.rows; ++i) {
            for (int j = 0; j < fullImage.cols; ++j) {
                Vec3b pixel = fullImage.at<Vec3b>(i, j); // BGR
                uchar gray = static_cast<uchar>((pixel[0] + pixel[1] + pixel[2]) / 3);
                grayImage.at<uchar>(i, j) = gray;
            }
        }
        rows = grayImage.rows;
        cols = grayImage.cols;

        kernel = generateHighPassKernel(kernelSize);
    }
    auto start_total = chrono::high_resolution_clock::now();
    // Broadcast image dimensions and kernel size
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast kernel data as a flat array
    vector<int> flatKernel;
    if (rank == 0) {
        for (const auto& row : kernel)
            flatKernel.insert(flatKernel.end(), row.begin(), row.end());
    }
    flatKernel.resize(kernelSize * kernelSize);
    MPI_Bcast(flatKernel.data(), kernelSize * kernelSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Reconstruct kernel locally
    if (rank != 0) {
        kernel.resize(kernelSize, vector<int>(kernelSize));
        for (int i = 0; i < kernelSize; ++i)
            for (int j = 0; j < kernelSize; ++j)
                kernel[i][j] = flatKernel[i * kernelSize + j];
    }

    int rowsPerProc = rows / size;
    int extra = (rank == size - 1) ? rows % size : 0;
    int localRows = rowsPerProc + extra;
    int half = kernelSize / 2;

    // Each process adds top/bottom padding rows
    int recvRows = localRows + 2 * half;

    Mat localImage(recvRows, cols, CV_8UC1);
    MPI_Datatype rowType;
    MPI_Type_contiguous(cols, MPI_UNSIGNED_CHAR, &rowType);
    MPI_Type_commit(&rowType);

    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int start = p * rowsPerProc;
            int end = (p == size - 1) ? rows : start + rowsPerProc;
            int sendRows = end - start;
            int sendStart = max(start - half, 0);
            int sendEnd = min(end + half, rows);
            int actualSendRows = sendEnd - sendStart;

            if (p == 0) {
                grayImage(Range(sendStart, sendEnd), Range::all()).copyTo(localImage);
            }
            else {
                MPI_Send(grayImage.ptr(sendStart), actualSendRows, rowType, p, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(localImage.data, recvRows, rowType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Apply kernel
    Mat localOutput(localRows, cols, CV_8UC1);
    for (int i = half; i < recvRows - half; ++i) {
        for (int j = 0; j < cols; ++j) {
            localOutput.at<uchar>(i - half, j) = applyKernel(localImage, kernel, i, j);
        }
    }

    // Gather all results to root
    Mat output;
    if (rank == 0) {
        output = Mat(rows, cols, CV_8UC1);
    }

    int* recvCounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recvCounts = new int[size];
        displs = new int[size];
        for (int p = 0; p < size; ++p) {
            int local = (p == size - 1) ? rows - (rowsPerProc * (size - 1)) : rowsPerProc;
            recvCounts[p] = local * cols;
            displs[p] = p * rowsPerProc * cols;
        }
    }

    MPI_Gatherv(localOutput.data, localRows * cols, MPI_UNSIGNED_CHAR,
        output.data, recvCounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Save and log time
    if (rank == 0) {
        auto end_total = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> total_time = end_total - start_total;

        imwrite("./outputs/HPF_MPI_outputRes.png", output);
        cout << "Image saved as output_mpi.png\n";
        cout << "Execution Time: " << total_time.count() << " ms\n";
        cout << " MPI size : " << size << endl;

        ofstream timeFile("./outputs/time.txt");
        timeFile <<"Execution Time: " << total_time.count() << endl;
        timeFile << " MPI size : " << size << endl;
        timeFile << "Kernal size used : " << kernelSize << endl;

        timeFile.close();
    }

    MPI_Type_free(&rowType);
    MPI_Finalize();
    return 0;
}
