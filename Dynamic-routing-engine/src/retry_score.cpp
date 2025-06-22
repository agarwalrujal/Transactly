#include "../include/retry_score.hpp"
#include <curl/curl.h>
#include <iostream>
#include <algorithm>
#include "../include/retry_score.hpp"

// Write callback to capture API response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* buffer) {
    buffer->append((char*)contents, size * nmemb);
    return size * nmemb;
}

float get_retry_score(const std::string& payload_json) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/predict");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_json.c_str());

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }

    // Manual parsing (quick and dirty)
    size_t pos = readBuffer.find(":");
    if (pos != std::string::npos) {
        std::string value_str = readBuffer.substr(pos + 1);
        value_str.erase(remove(value_str.begin(), value_str.end(), '}'), value_str.end());
        return std::stof(value_str);
    }

    return -1.0f;
}
