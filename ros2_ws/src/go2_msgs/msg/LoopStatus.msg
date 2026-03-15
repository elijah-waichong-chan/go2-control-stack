int32 status

# Average loop time in milliseconds.
float32 avg_loop_ms

# 99th percentile loop time in milliseconds.
float32 p99_loop_ms

# Maximum loop time in milliseconds.
float32 max_loop_ms

# Loop budget in milliseconds.
float32 budget_ms

# Number of loop iterations that exceeded the budget.
int32 deadline_miss_count

# Number of loop samples included in the statistics.
int32 sample_count
