#include "compliance.hpp"
#include "json/json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>

using json = nlohmann::json;

static std::unordered_set<std::string> blocked_corridors;

void load_compliance_rules() {
    std::ifstream in("compliance_rules.json");
    if (!in.is_open()) {
        std::cerr << "Error opening compliance_rules.json\n";
        return;
    }

    json j;
    in >> j;

    for (const auto& corridor : j["blocked_corridors"]) {
        blocked_corridors.insert(corridor.get<std::string>());
    }

    std::cout << "Loaded compliance rules. Blocked: " << blocked_corridors.size() << " corridors.\n";
}

bool is_compliant(const Corridor& corridor) {
    std::string key = corridor.from + "-" + corridor.to;
    return blocked_corridors.find(key) == blocked_corridors.end();
}
