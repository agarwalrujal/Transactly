# Transactly
Transactly is a full-stack intelligent payment routing and fraud detection platform that simulates a real-time cross-border payment ecosystem. It leverages AWS services, AI/ML models, and C++/Python microservices to compute the most optimal and compliant routes for financial transactions.


#DYNMAIC ROUTING ENGINE
The Dynamic Routing Engine is a C++-based system that computes the most optimal top-K transaction paths across global payment corridors. It leverages an A*-based search algorithm, integrates real-time ML retry scoring via HTTP from a Python Flask server, and enforces regional compliance filters.

This engine is core to the intelligent path selection in the Transactly platform, simulating how a real-world fintech might route cross-border transactions optimally, securely, and intelligently.

->router.cpp
      Purpose: Implements the A* search algorithm to find top-K most optimal paths between source and destination nodes in a payment graph
      Functions:-
        load_graph(): Initializes a simulated corridor graph with latency and cost attributes.
        get_top_k_routes(source, target, K): Uses A* logic to explore the best K paths using weighted latency and cost
        get_cost(corridor): Combines latency and cost using a linear weight formula.
        get_route(source, goal): Filters paths using compliance and dynamic retry scoring via ML.
      
->retry_score.cpp
      Purpose: Makes HTTP POST requests from C++ to a local Python Flask server to get ML-generated retry scores.
      
->compliance.hpp
      Purpose: Contains hardcoded or configurable compliance rules that filter out disallowed corridors (e.g., embargoed countries).
JSON payload:
  {
    "corridor_id": "A_B",
    "status": "active",
    "corridor_type": "primary",
    "success_rate_7d": random(0,1),
    "latency_ms": 180,
    "cost_score": 0.10,
    "past_retry_success_rate": random(0,1)
  }
GRAPH USED HERE:-
    IN
    ├── SG ── US
    │        └── IR ✖️ (non-compliant)
    ├── AE ── US
    │     └── EU ── US
    ├── RU ── KP ── US
    
  Testing & Validation
    Run Flask ML Server:
          python app.py
    Build & Run C++ Server:
        g++ -std=c++17 main.cpp -lcurl
        ./Microservice.out
    Query the Microservice:
      start node=start
      end node=end
        curl "http://localhost:8080/getRoute?start='start'&goal='end'"
      example:
        curl "http://localhost:8080/getRoute?start=IN&goal=US"

