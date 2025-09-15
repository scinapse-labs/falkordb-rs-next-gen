# Profiling using pyroscope

To profile the code to find performance bottelnecks we use pyroscope

1. Compile the code using the pyro feature
   ```
   cargo build -F pyro
   ```

2. Run the pyroscope docker container
   ```
    docker network create pyroscope-demo
    docker run --rm --name pyroscope --network=pyroscope-demo -p 4040:4040 grafana/pyroscope:latest
    ```

3. Run the database and your query

4. Navigate to the pyroscope UI at https://localhost:4040

5. use the falkordb application