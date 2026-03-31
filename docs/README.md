Real-time Recommendation System Using the Two-Tower Architecture
Overview:

A dedicated book recommendation system based on the Two-Tower architecture, using a combination of FastAPI, Redis, and Faiss to deliver fast real-time recommendations.

The system is designed to process user data in real time, with the ability to handle the Cold Start problem and provide accurate recommendations based on interaction history.

- Technical Components:
Two-Tower Model: used to generate embeddings for users and items.
FastAPI: used to provide a fast and efficient API.
Redis: used to store user vectors and interaction history in real time.
Faiss: used for fast nearest-neighbor search in vector space.
Docker: used to containerize the system and ensure easy deployment and execution.

-Key Features:
Real-time updating of user vectors at every recommendation request.
Personalized recommendations based on the user’s activity history.
Handling the Cold Start problem by:
suggesting the most interacted-with items for new users.
gradually improving recommendations after the first interactions are collected.
High-performance API built with FastAPI.
Fast data storage using Redis to reduce latency.
Using Faiss to provide fast recommendations even with large-scale data.
Containerization support through Docker to simplify deployment and execution.

- Architecture Overview:
Users and items are represented using the Two-Tower model.
User vectors are stored in Redis.
At every request:
the user vector is updated based on the user’s interactions;
the closest items are retrieved using Faiss.

For a new user:
popular items are presented.
the system gradually shifts to personalized recommendations once enough data becomes available.

- Workflow:
Receive a recommendation request from the user through the API.
Check whether the user has a previous history.
Update the user embedding in Redis.
Query Faiss to retrieve the closest items.
Return the recommendation list.

- Running with Docker:
The system can be run easily through Docker to ensure a consistent environment:
build the image;
run the container;
access the API through FastAPI.

- Project Goal:
Build a production-ready recommendation system.
Apply deep learning techniques to recommendation.
Address real-time system challenges.
Improve performance using approximate nearest-neighbor search techniques.
