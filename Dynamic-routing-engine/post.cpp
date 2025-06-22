#include <iostream>
#include <string>
#include <curl/curl.h>
#include "include/json.hpp" // nlohmann/json header

using json = nlohmann::json;

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

int main() {
    CURL* curl;
    CURLcode res;

    std::string readBuffer;

    std::string jsonData = R"({
        "corridor_id": "IN_EU",
        "status": "active",
        "corridor_type": "primary",
        "success_rate_7d": 0.91,
        "latency_ms": 180,
        "cost_score": 0.10,
        "past_retry_success_rate": 0.97
    })";

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/predict");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL Error: " << curl_easy_strerror(res) << std::endl;
        } else {
            try {
                json response_json = json::parse(readBuffer);
                float score = response_json["retry_success_probability"];

                std::cout << "Prediction Score: " << score << std::endl;

                // ✅ Use the score in real-time decision logic
                if (score > 0.7) {
                    std::cout << "🟢 Proceed with retry!" << std::endl;
                    // Insert retry logic
                } else {
                    std::cout << "🔴 Skip retry." << std::endl;
                    // Insert fallback logic
                }

            } catch (json::exception& e) {
                std::cerr << "JSON Parsing error: " << e.what() << std::endl;
            }
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    curl_global_cleanup();
    return 0;
}
