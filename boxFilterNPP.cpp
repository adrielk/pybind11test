/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//Adriel's To do:
/*
    1. Pack the convolution into a neat function with paramters: filename, kernel, image, etc..
    2. Find a way to bind this to python. Can discuss this. I can only find C++ and python binders? 
    3. What is oAnchor???(Seems to just offset the image sightly, be wary)
    4. Play with ROI
    5. TIME THIS CODE! (DONE)
    6. ISSUE: Kernel is Npp32f not Npp32f (we need it to be a float!) (DONE)
    7. ISSUE: There is a problem with your bfKernel, i suppose it might need some normalization?? It just makes everything black...
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

//Additional includes
#include <sys/time.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <typeinfo>
#include <stdlib.h>

long long start_timer();
long long stop_timer(long long start_time, const char* name);
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize);


inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}
//NORMALIZE THE KERNEL (That's what makes it visible to us, although I'm not sure if this is done in the LSST codebase! (or maybe that do it differently))
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize) {
    std::fstream file;
    std::string word, t, q, filename;
    // filename of the file
    filename = kernelName;

    // opening file
    file.open(filename.c_str());
    double sum = 0;

    // extracting words from the file
    if (file.is_open()) {
        for (int i = 0; i < kernelSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            double wordDouble = strtod(char_array, &pEnd);
            kernelArr[i] = wordDouble;
            sum += wordDouble;
            /*if (i == 144)//testing out identity kernel.
                kernelArr[i] = 1;//wordDouble;
            else
                kernelArr[i] = 0;*/
        }

    }

    for (int i = 0;i < kernelSize;i++) {
        kernelArr[i] = kernelArr[i] / sum;
    }

    file.close();
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char* filePath;

        cudaDeviceInit(argc, (const char**)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char**)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char**)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_boxFilter.pgm";

        if (checkCmdLineFlag(argc, (const char**)argv, "output"))
        {
            char* outputFilePath;
            getCmdLineArgumentString(argc, (const char**)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }
        long long start_time = start_timer();

        //Code from stack overflow - begin
        std::string fileExtension = ".pgm";
        std::string dirFilename = "lena";
        std::string saveFilename = dirFilename + "_convolved";
        //npp::ImageCPU_8u_C1 oHostSrc;
        npp::ImageCPU_32f_C1 oHostSrc;
       
        //npp::loadImage(dirFilename + fileExtension, oHostSrc);//(sFilename, oHostSrc);
        //npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc); // malloc and memcpy to GPU 
        npp::ImageNPP_32f_C1 oDeviceSrc(512,512);//(oHostSrc.width(), oHostSrc.height());
        NppiSize kernelSize = { 17,17 };//{ 3, 3 }; // dimensions of convolution kernel (filter)
        NppiSize oSizeROI = {510,510 };//oHostSrc.width() - kernelSize.width + 1, oHostSrc.height() - kernelSize.height + 1 };//what is with the kernel offset of ROI? How does this deal with the edges? Avoiding them?
        //npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
        //npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());

        npp::ImageNPP_32f_C1 oDeviceDst(510,510);//oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
        npp::ImageCPU_32f_C1 oHostDst(oDeviceDst.size());
        NppiPoint oAnchor = { kernelSize.width / 2, kernelSize.height / 2 }; //**SEE DOCUMENTATION ON WHAT oAnchor is??? (Does this perfectly center it? Im just copying example code)
        NppStatus eStatusNPP;

        std::cout << "kernelSize.width: " << kernelSize.width / 2 << "/kernelSize.height: " << kernelSize.height / 2 << std::endl;
        int hostKSize = kernelSize.width * kernelSize.height;
        //Npp32f hostKernel[hostKSize];//= { 0, -1, 0, -1, 5, -1, 0, -1, 0 };//{ 0,0,0,0,1,0,0,0,0 };//Identity kernel to test alignment//{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };//this is emboss//{ -1, 0, 1, -1, 0, 1, -1, 0, 1 }; // convolving with this should do edge detection
        Npp32f hostKernel[hostKSize];
        fillKernelArray("bfKernel.txt", hostKernel, hostKSize);

        /*
        for (int i = 0;i < 289;i++) {
            hostKernel[i] = 1/9; //a blur kernel...?
        }*/
        
        /*for (int i = 0; i < hostKSize;i++) {
            std::cout << hostKernel[i] << std::endl;
        }*/

        Npp32f* deviceKernel;
        size_t deviceKernelPitch;
        //cudaMallocPitch((void**)&deviceKernel, &deviceKernelPitch, kernelSize.width * sizeof(Npp32f), kernelSize.height * sizeof(Npp32f));
        /*cudaMemcpy2D(deviceKernel, deviceKernelPitch, hostKernel,
            sizeof(Npp32f) * kernelSize.width, // sPitch
            sizeof(Npp32f) * kernelSize.width, // width
            kernelSize.height, // height
            cudaMemcpyHostToDevice);*/
        cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f));//is Npp32f 32 bit? We may be 64 bit in the future!
        cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f), cudaMemcpyHostToDevice);
        Npp32f divisor = 1; // no scaling

        std::cout << "Calculated size: " << kernelSize.width * kernelSize.height * sizeof(Npp32f) << std::endl;
        std::cout << "Device kernel size: " << sizeof(deviceKernel) << std::endl;
        std::cout << "hostKernel size: " << sizeof(hostKernel) << std::endl;

        //eStatusNPP = nppiFilter32f_8u_C1R(oDeviceSrc.data());
        int devPitch = oDeviceSrc.pitch();
        int dstPitch = oDeviceDst.pitch();
        //std::cout <<"Pitch:" <<oDeviceSrc.pitch() << std::endl;
        //How pitch is calculated: how many bytes in a row? Calcualte by getting bytes in a pixel * image width
        
        //std::cout << "Source image: " << oDeviceSrc.data() << std::endl;
        std::cout << "Source image Line Step (bytes) " << devPitch << std::endl;
        //std::cout << "Destination Image: " << oDeviceDst.data() << std::endl;
        std::cout << "Destination Image line step (bytes): " << dstPitch << std::endl;
        //std::cout << "ROI: " << oSizeROI << std::endl;
        //std::cout << "Device Kernel: " << deviceKernel << std::endl;
        //std::cout << "Kernel Size: " << kernelSize << std::endl;
        //std::cout << "X and Y offsets of kernel origin frame: " << oAnchor << std::endl;

        eStatusNPP = nppiFilter_32f_C1R(oDeviceSrc.data(), devPitch, oDeviceDst.data(),
            dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor);
        /*eStatusNPP = nppiFilter_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, deviceKernel, kernelSize, oAnchor, divisor);*/

        std::cout << "NppiFilter error status " << eStatusNPP << std::endl; // prints 0 (no errors)
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch()); // memcpy to host
        //saveImage(saveFilename+fileExtension, oHostDst);

        /*Npp8u* hostDstData = oHostDst.data();
        std::cout << "Host destination data type:  " << typeid(hostDstData).name() << std::endl; //unsigned char array of course...
        for (int i = 0;i < hostDstData.size();i++) {
            std::cout << hostDstData[i] << std::endl;
        }*/
        for (int i = 0;i < 512 * 512;i++) { 
           std::cout << oHostDst.data()[i] << std::endl;
        }        //end code from SO
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
    
        long long totalTime = stop_timer(start_time, "Total NPP convolution time:");
        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time) / (1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char* name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}
