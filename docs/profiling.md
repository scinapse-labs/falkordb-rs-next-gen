# Profiling using samply

To profile the code to find performance bottlenecks we use [samply](https://github.com/mstange/samply).

1. Install samply
   ```
   cargo install samply
   ```

2. Build the code in release mode with debug info
   ```
   cargo build --release
   ```

3. Run the database under samply
   ```
   samply record redis-server --loadmodule ./target/release/libfalkordb.dylib
   ```

4. Run your query against the database

5. Stop the database and samply will open the Firefox Profiler UI in your browser
