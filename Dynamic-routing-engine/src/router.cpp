#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <sstream>
#include <limits>
#include <algorithm>
#include <random>
#include <iomanip>
#include <sstream>
#include "retry_score.cpp"
std::string build_payload_json(const std::string& from, const std::string& to, double latency, double cost) {
    // Generate random floats between 0.0 and 1.0
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0.0, 1.0);

    float success_rate_7d = dist(gen);
    float past_retry_success_rate = dist(gen);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << R"({
        "corridor_id": ")" << from << "_" << to << R"(",
        "status": "active",
        "corridor_type": "primary",
        "success_rate_7d": )" << success_rate_7d << R"(,
        "latency_ms": )" << latency << R"(,
        "cost_score": )" << cost << R"(,
        "past_retry_success_rate": )" << past_retry_success_rate << "\n"
        << "})";
        std::cout << oss.str() << std::endl;
        std::string input_json = R"({
        "corridor_id": "IN_EU",
        "status": "active",
        "corridor_type": "primary",
        "success_rate_7d": 0.91,
        "latency_ms": 180,
        "cost_score": 0.10,
        "past_retry_success_rate": 0.97
    })";
    return input_json;
}

// Corridor structure
struct Corridor {
    std::string from;
    std::string to;
    double latency_ms;
    double cost_score;

    Corridor(std::string f, std::string t, double l, double c)
        : from(f), to(t), latency_ms(l), cost_score(c) {}
};

// Global graph
std::unordered_map<std::string, std::vector<Corridor>> graph;

// Compliance checker: blocks corridors to certain countries
bool is_compliant(const Corridor& corridor) {
    static std::unordered_set<std::string> blocked_countries = {};
    return blocked_countries.find(corridor.to) == blocked_countries.end();
}

// Weighted cost function
double get_cost(const Corridor& c) {
    double alpha = 1.0;     // weight for latency
    double beta = 100.0;    // weight for cost_score
    return alpha * c.latency_ms + beta * c.cost_score;
}

// Heuristic function — set to 0 (Dijkstra-like behavior)
double heuristic(const std::string& current, const std::string& goal) {
    return 0.0;
}

// A* function to get top K compliant routes
std::vector<std::pair<std::vector<std::string>, double>> get_top_k_routes(const std::string& start, const std::string& goal, int K) {
    using PQItem = std::pair<double, std::vector<std::string>>; // <estimated total cost, path>
    auto cmp = [](const PQItem& a, const PQItem& b) {
        return a.first > b.first;
    };
    std::priority_queue<PQItem, std::vector<PQItem>, decltype(cmp)> open_set(cmp);

    std::vector<std::pair<std::vector<std::string>, double>> results;

    open_set.push({0.0, {start}});

    while (!open_set.empty() && results.size() < K) {
        //auto &[f_cost, path] = open_set.top();
        double f_cost = open_set.top().first;
        std::vector<std::string> path = open_set.top().second;
        open_set.pop();
        std::string current = path.back();

        if (current == goal) {
            results.push_back({path, f_cost});
            continue;
        }

        for (const auto& corridor : graph[current]) {
            //if (!is_compliant(corridor)) continue;  // 🚫 skip non-compliant corridors

            std::vector<std::string> new_path = path;
            new_path.push_back(corridor.to);

            double g_cost = f_cost - heuristic(current, goal) + get_cost(corridor);
            double f_new = g_cost + heuristic(corridor.to, goal);

            open_set.push({f_new, new_path});
        }
    }

    return results;
}

// Graph initialization
void load_graph() {
    graph["IN"] = {
        {"IN", "SG", 100, 0.05},
        {"IN", "RU", 300, 0.20},
        {"IN", "AE", 80, 0.02}
    };

    graph["SG"] = {
        {"SG", "US", 200, 0.07},
        {"SG", "IR", 350, 0.15}   // 🚫 blocked
    };

    graph["AE"] = {
        {"AE", "US", 150, 0.06},
        {"AE", "EU", 90, 0.05}
    };

    graph["RU"] = {
        {"RU", "KP", 400, 0.25}   // 🚫 blocked
    };

    graph["IR"] = {
        {"IR", "SY", 3000, 0.12}   // 🚫 blocked
    };

    graph["EU"] = {
        {"EU", "US", 180, 0.03}
    };

    graph["KP"] = {
        {"KP", "US", 500, 0.30}
    };

    graph["SY"] = {
        {"SY", "US", 320, 0.10}
    };
}

// Print paths
void print_paths(const std::vector<std::pair<std::vector<std::string>, double>>& paths) {
    int count = 1;
    for (const auto& [path, cost] : paths) {
        std::cout << "Path " << count++ << ": ";
        for (const auto& node : path) std::cout << node << " ";
        std::cout << "| Total Cost: " << cost << "\n";
    }
}

// Main
std::vector<std::pair<std::vector<std::string>,double>> get_route(std::string &start, std::string &goal) {
    load_graph();

    std::string source = start;
    std::string destination = goal;
    int K = 5;
    std::vector<std::pair<std::vector<std::string>,double>> top_routes; 
    auto top_paths = get_top_k_routes(source, destination, K);
    for (const auto& path : top_paths) {
        const std::vector<std::string>& nodes = path.first;
        double total_score = 1.0;

        for (size_t i = 0; i < nodes.size() - 1; ++i) {
            const std::string& from = nodes[i];
            const std::string& to = nodes[i + 1];

            // Find corridor
            auto it = std::find_if(graph[from].begin(), graph[from].end(), [&](const Corridor& c) {
                return c.to == to;
            });

            if (it != graph[from].end()) {
                std::string payload = build_payload_json(from, to, it->latency_ms, it->cost_score);
                float score = get_retry_score(payload);
                //std::cout << "Score for " << from << " → " << to << ": " << score << "\n";
                total_score *= score;
            }
        }
        std::pair<std::vector<std::string>,double> x = {nodes, total_score};
        top_routes.push_back(x);

        std::cout << "Total retry score for path: " << total_score << "\n\n";
    }

    // if (top_paths.empty()) {
    //     std::cout << "No compliant path found.\n";
    // } else {
    //     print_paths(top_paths);
    // }
    std::string s="abs";
    return top_routes;
}