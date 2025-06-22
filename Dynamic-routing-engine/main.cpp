#include "libs/httplib.h"
//#include "include/compliance.hpp"
//#include "include/router.hpp"
//#include "include/retry_score.hpp"
#include "src/router.cpp"
#include <iostream>

int main() {

    httplib::Server svr;
    // load_compliance_rules();
    // load_graph();
    svr.Get("/getRoute", [](const httplib::Request& req, httplib::Response& res) {
        std::string from = req.get_param_value("start");
        std::string to = req.get_param_value("goal");
        // Example: Add logging
    std::cout << "[RouteRequest] from=" << from << " to=" << to << std::endl;

        std::vector<std::pair<std::vector<std::string>,double>> result = get_route(from, to);
        std::string result_str;
        for (const auto& route : result) {
            result_str += "Path: ";
            for (const auto& node : route.first) {
                result_str += node + " ";
            }
            result_str += "| Total Score: " + std::to_string(route.second) + "\n";
        }
        if (result.empty()) {
            result_str = "No compliant path found.";
        }
        res.set_content(result_str, "text/plain");

    });

    std::cout << "Server started on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);
    return 0;
}
