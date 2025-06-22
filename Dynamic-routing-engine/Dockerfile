FROM gcc:13

# Create app directory
WORKDIR /app

# Install dependencies (e.g. for JSON and HTTP server)
RUN apt-get update && apt-get install -y cmake

# Copy source files
COPY . .

# Build the project
RUN cmake . && make

# Expose the port your server listens on
EXPOSE 8080

# Run the compiled binary
CMD ["./dynamic_router"]
