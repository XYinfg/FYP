# analysis.py
import pickle
import os
import matplotlib.pyplot as plt
from metrics import summarize_results

os.makedirs("./results/figures", exist_ok=True)

with open("./results/outputs/stratus_sim.pkl", "rb") as f:
    stratus_sim = pickle.load(f)
with open("./results/outputs/fcfs_sim.pkl", "rb") as f:
    fcfs_sim = pickle.load(f)
with open("./results/outputs/minmin_sim.pkl", "rb") as f:
    minmin_sim = pickle.load(f)

stratus_results = summarize_results(stratus_sim)
fcfs_results = summarize_results(fcfs_sim)
minmin_results = summarize_results(minmin_sim)

print("Algorithm Comparison:")
print("{:<10} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
    "Algorithm", "Avg Completion", "Median Completion", "Fairness", "Utilization", "Cost"
))
for name, res in [("Stratus", stratus_results), ("FCFS", fcfs_results), ("MinMin", minmin_results)]:
    print("{:<10} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(
        name, 
        res["avg_completion_time"] if res["avg_completion_time"] is not None else float('nan'),
        res["median_completion_time"] if res["median_completion_time"] is not None else float('nan'),
        res["fairness"] if res["fairness"] is not None else float('nan'),
        res["utilization"] if res["utilization"] is not None else float('nan'),
        res["cost"] if res["cost"] is not None else float('nan')
    ))

algorithms = ["Stratus", "FCFS", "MinMin"]
avg_ct = [stratus_results["avg_completion_time"], fcfs_results["avg_completion_time"], minmin_results["avg_completion_time"]]
fairness = [stratus_results["fairness"], fcfs_results["fairness"], minmin_results["fairness"]]
utilization = [stratus_results["utilization"], fcfs_results["utilization"], minmin_results["utilization"]]
cost = [stratus_results["cost"], fcfs_results["cost"], minmin_results["cost"]]

# Bar chart for Average Completion Time
plt.figure(figsize=(6,4))
plt.bar(algorithms, avg_ct, color=['blue','red','green'])
plt.title("Average Completion Time Comparison")
plt.ylabel("Time (seconds)")
plt.savefig("./results/figures/avg_completion_time.png")
plt.close()

# Bar chart for Fairness
plt.figure(figsize=(6,4))
plt.bar(algorithms, fairness, color=['blue','red','green'])
plt.title("Fairness Comparison (Jain's Index)")
plt.ylabel("Fairness (dimensionless)")
plt.savefig("./results/figures/fairness.png")
plt.close()

# Bar chart for Utilization
plt.figure(figsize=(6,4))
plt.bar(algorithms, utilization, color=['blue','red','green'])
plt.title("Resource Utilization Comparison")
plt.ylabel("CPU Utilization (approx)")
plt.savefig("./results/figures/utilization.png")
plt.close()

# Bar chart for Cost
plt.figure(figsize=(6,4))
plt.bar(algorithms, cost, color=['blue','red','green'])
plt.title("Cost Comparison")
plt.ylabel("Cost (arbitrary units)")
plt.savefig("./results/figures/cost.png")
plt.close()

print("Analysis complete. Check ./results/figures for visualization outputs.")
