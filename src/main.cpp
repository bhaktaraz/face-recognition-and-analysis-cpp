#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <ctime>
#include <vector>
#include <curl/curl.h>
#include <string>
#include <nlohmann/json.hpp> // Include nlohmann/json library

using json = nlohmann::json;
using namespace std;
using namespace cv;

bool captureImage = false;
Mat capturedImage;

static const char *base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len)
{
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--)
    {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;
}

// Helper function to encode the image to base64
string image_to_base64(const cv::Mat &img)
{
    vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
    string encoded = base64_encode(enc_msg, buf.size());
    return encoded;
}

// Function to handle the HTTP response
size_t write_callback(void *contents, size_t size, size_t nmemb, string *s)
{
    size_t newLength = size * nmemb;
    try
    {
        s->append((char *)contents, newLength);
    }
    catch (std::bad_alloc &e)
    {
        // Handle memory problem
        return 0;
    }
    return newLength;
}

// Function to send image to server and receive detected emotion
string detectEmotion(const cv::Mat &faceImage)
{
    CURL *curl;
    CURLcode res;
    string response_string;
    string header_string;
    string encoded_image = image_to_base64(faceImage);

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl)
    {
        string url = "http://localhost:3000/recognize";
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        json j;
        j["imageBuffer"] = encoded_image;
        string postData = j.dump();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        curl_global_cleanup();
    }
    std::cout << "Value: " << response_string << "\nType: " << typeid(response_string).name() << std::endl;

    return response_string;
}

bool compareFacesUsingHistogram(const cv::Mat &face1, const cv::Mat &face2)
{
    cv::Mat hist1, hist2;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};

    // Convert to grayscale only if they are not already
    cv::Mat gray1, gray2;
    if (face1.channels() == 3)
    {
        cvtColor(face1, gray1, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray1 = face1.clone();
    }

    if (face2.channels() == 3)
    {
        cvtColor(face2, gray2, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray2 = face2.clone();
    }

    // Calculate the histograms
    calcHist(&gray1, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange, true, false);
    calcHist(&gray2, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange, true, false);

    // Normalize the histograms
    normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    // Compare using the Chi-Square method
    double result = compareHist(hist1, hist2, cv::HISTCMP_CHISQR);
    std::cout << "Histogram similarity: " << result << std::endl;

    // Define a threshold for deciding if the images are similar
    double similarityThreshold = 0.1; // Adjust this threshold based on your testing
    return result < similarityThreshold;
}

cv::Rect detectLargestFace(cv::Mat &img, cv::CascadeClassifier &face_cascade)
{
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(img, faces, 1.3, 5);
    if (faces.empty())
    {
        return cv::Rect(); // Return an empty rect if no faces are found
    }
    // Assuming the largest face is the most relevant one
    return *std::max_element(faces.begin(), faces.end(),
                             [](const cv::Rect &a, const cv::Rect &b)
                             {
                                 return a.area() < b.area();
                             });
}

bool compareImages(const cv::Mat &img1, const cv::Mat &img2)
{
    // Initialize ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Match descriptors using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // Check if enough matches are found
    const int MIN_MATCH_COUNT = 30; // Set a threshold for considering a match
    if (good_matches.size() >= MIN_MATCH_COUNT)
    {
        // Find homography using RANSAC
        std::vector<cv::Point2f> pts1, pts2;
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
            pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }
        cv::Mat mask;
        cv::findHomography(pts1, pts2, cv::RANSAC, 5.0, mask); // 5.0 is the RANSAC reprojection threshold

        // Count inliers (matches that are geometrically consistent)
        int inliers = 0;
        for (int i = 0; i < mask.rows; ++i)
        {
            if (mask.at<uchar>(i))
            {
                inliers++;
            }
        }

        // If enough inliers are found, then the images are a match
        if (inliers >= MIN_MATCH_COUNT)
        {
            return true;
        }
    }
    return false;
}

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        captureImage = true;
    }
}

void detectEyes(Mat &img, Rect face, CascadeClassifier &eyes_cascade)
{
    Mat ROI = img(face);
    vector<Rect> eyes;
    eyes_cascade.detectMultiScale(ROI, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const Rect &e : eyes)
    {
        Rect eyeRect(face.x + e.x, face.y + e.y, e.width, e.height);
        rectangle(img, eyeRect.tl(), eyeRect.br(), Scalar(0, 255, 0), 2);
    }
}

void detectSmile(Mat &img, Rect face, CascadeClassifier &smile_cascade)
{
    Mat ROI = img(face);
    vector<Rect> smiles;
    smile_cascade.detectMultiScale(ROI, smiles, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const Rect &smile : smiles)
    {
        Point center(face.x + smile.x + smile.width / 2, face.y + smile.y + smile.height / 2);
        ellipse(img, center, Size(smile.width / 2, smile.height / 2), 0, 0, 360, Scalar(255, 0, 0), 4);
    }
}

int main()
{

    VideoCapture video(0);
    namedWindow("Camera", cv::WINDOW_NORMAL);
    CascadeClassifier facedetect, eyes_cascade, smile_cascade;

    if (!video.isOpened())
    {
        std::cout << "Error: Couldn't open the camera." << endl;
        return -1;
    }

    setMouseCallback("Camera", onMouse, NULL);

    if (!facedetect.load("C:/Users/Bibek Joshi/Desktop/ImageRecognition/data/haarcascades/lbpcascade_frontalface_improved.xml") ||
        !eyes_cascade.load("C:/Users/Bibek Joshi/Desktop/ImageRecognition/data/haarcascades/haarcascade_eye.xml") ||
        !smile_cascade.load("C:/Users/Bibek Joshi/Desktop/ImageRecognition/data/haarcascades/haarcascade_smile.xml"))
    {
        std::cout << "Error: Couldn't load one or more cascade classifiers." << endl;
        return -1;
    }

    Mat img;
    ofstream csvFile("C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/data.csv", ios::app);

    cv::Mat bibekImage = cv::imread("C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/bibek.jpg", cv::IMREAD_GRAYSCALE);

    while (true)
    {
        video.read(img);

        // Check if the frame is empty
        if (img.empty())
        {
            std::cout << "Error: Couldn't read frame from the camera." << endl;
            break;
        }

        if (captureImage)
        {
            // Copy the image for display in the prompt window
            capturedImage = img.clone();

            // Get the current time
            std::time_t now = std::time(nullptr);

            // Convert to local time
            std::tm ltm;
            localtime_s(&ltm, &now); // Correct usage of localtime_s

            // Create a timestamp string
            char timestamp[20];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &ltm);

            // Save image
            string imageName = "C:/Users/Bibek Joshi/Desktop/ImageRecognition/images/captured_photo_" + string(timestamp) + ".jpg";
            imwrite(imageName, capturedImage);
            std::cout << "Photo captured and saved as " << imageName << endl;
            // Reset the capture flag
            captureImage = false;

            // Call the Emotion Detect Function
            std::string rawJsonResponse = detectEmotion(capturedImage);

            // Parse the JSON response
            json jsonResponse = json::parse(rawJsonResponse);

            // Access the values from the parsed JSON
            std::string detectedEmotion = jsonResponse["emotion"];
            double emotionScore = jsonResponse["score"];

            // Show the Detected Emotion in New Window
            cv::namedWindow("Emotion Info", cv::WINDOW_AUTOSIZE);
            cv::putText(capturedImage, "Detected Emotion: " + detectedEmotion, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(capturedImage, "Score: " + std::to_string(emotionScore), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Emotion Info", capturedImage);

            cv::waitKey(0);

            // cv::Rect largestFaceInCaptured = detectLargestFace(capturedImage, facedetect);
            // cv::Rect largestFaceInBibek = detectLargestFace(bibekImage, facedetect);

            // if (!largestFaceInCaptured.empty() && !largestFaceInBibek.empty())
            // {
            //     // Crop the detected faces
            //     cv::Mat faceCaptured = capturedImage(largestFaceInCaptured);
            //     cv::Mat faceBibek = bibekImage(largestFaceInBibek);
            //     bool isMatch = compareFacesUsingHistogram(faceCaptured, faceBibek);
            //     std::cout << "Image Compare: " << (isMatch ? "Match found." : "No match found.") << std::endl;
            // }
            // else
            // {
            //     std::cout << "Face not detected in one or both images." << std::endl;
            // }

            // if (compareImages(capturedImage, bibekImage))
            // {
            //     // If a match is found, display the image in a new window
            //     cv::namedWindow("Matched Image", cv::WINDOW_AUTOSIZE);
            //     cv::imshow("Matched Image", bibekImage);
            //     cv::waitKey(0); // Wait for a key press to close the window
            // }
            // else
            // {
            //     cv::namedWindow("No Match Found", cv::WINDOW_AUTOSIZE);
            //     cv::imshow("No Match Found", capturedImage);
            //     cv::waitKey(0); // Wait for a key press to close the window
            // }
        }

        // Face Detecting Code For multiple faces
        vector<Rect> faces;
        facedetect.detectMultiScale(img, faces, 1.3, 5);

        std::cout << faces.size() << " face(s) found." << endl;

        for (const Rect &face : faces)
        {
            // Use the rectangle function from the imgproc namespace
            rectangle(img, face.tl(), face.br(), Scalar(50, 50, 255), 2);
            cv::rectangle(img, Point(0, 0), Point(250, 70), Scalar(50, 50, 255), FILLED);
            putText(img, to_string(faces.size()) + " Face Found", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
            // Draw facial features
            detectEyes(img, face, eyes_cascade);
            detectSmile(img, face, smile_cascade);
        }

        imshow("Camera", img);

        // Wait for a short time and check if the user has closed the window
        if (cv::waitKey(1) == 27 || cv::getWindowProperty("Camera", cv::WND_PROP_VISIBLE) < 1)
        {
            break; // Break the loop if the user presses 'Esc' or closes the window
        }
    }

    // After breaking out of the loop, release the VideoCapture object and destroy all windows
    video.release();
    cv::destroyAllWindows();

    csvFile.close();
    return 0;
}
