# Transactly
Transactly is a full-stack intelligent payment routing and fraud detection platform that simulates a real-time cross-border payment ecosystem. It leverages AWS services, AI/ML models, and C++/Python microservices to compute the most optimal and compliant routes for financial transactions.


#DYNAMIC ROUTING ENGINE
The Dynamic Routing Engine is a C++-based system that computes the most optimal top-K transaction paths across global payment corridors. It leverages an A*-based search algorithm, integrates real-time ML retry scoring via HTTP from a Python Flask server, and enforces regional compliance filters.

This engine is core to the intelligent path selection in the Transactly platform, simulating how a real-world fintech might route cross-border transactions optimally, securely, and intelligently.



