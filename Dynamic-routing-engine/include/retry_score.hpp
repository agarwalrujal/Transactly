#ifndef RETRY_SCORE_H
#define RETRY_SCORE_H

#include <string>

// Fetch retry score from Flask API for given JSON input
float get_retry_score(const std::string& payload_json);

#endif // RETRY_SCORE_H
