#pragma once
#include <string>

// struct Corridor {
//     std::string from;
//     std::string to;
//     double latency_ms;
//     double cost_score;

//     Corridor(std::string f, std::string t, double l, double c)
//         : from(f), to(t), latency_ms(l), cost_score(c) {}
// };

void load_graph();
std::string get_route(const std::string& start, const std::string& goal);
