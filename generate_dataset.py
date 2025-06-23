import pandas as pd
import random

# --- Configuration ---
NUM_SAMPLES = 200
FILENAME = "my_data.csv"

# Define the data templates. Each class has a distinct theme.
data_templates = {
    'A': [
        "Log entry: User authentication failed. Error code: 401.",
        "System report: CPU usage is at 95%. Scaling up resources.",
        "New commit pushed to main branch. CI/CD pipeline initiated.",
        "Database query executed successfully. Latency: 45ms.",
        "Security alert: Unusual login attempt detected from new IP.",
    ],
    'B': [
        "Q4 financial report shows a 15% increase in revenue.",
        "Market analysis: consumer confidence is trending upwards.",
        "Shareholder meeting scheduled for next Tuesday to discuss acquisition.",
        "New marketing campaign to launch in the EMEA region next month.",
        "Supply chain logistics have been optimized, reducing costs by 8%.",
    ],
    'C': [
        "Study finds new species of deep-sea microbe near hydrothermal vents.",
        "Particle accelerator experiment confirms theoretical particle's existence.",
        "Climate model predicts a 1.5°C temperature rise by 2050.",
        "Photosynthesis rates observed to increase with higher CO2 levels.",
        "Astronomers have identified a new exoplanet within the habitable zone.",
    ],
    'D': [
        "Literary review praises the novel's use of post-modern narrative techniques.",
        "The art exhibition features cubist paintings from the early 20th century.",
        "Symphony orchestra announces its schedule for the upcoming season.",
        "Analysis of the film highlights its groundbreaking cinematography.",
        "The poem's central theme revolves around loss and redemption.",
    ],
}

# --- Data Generation ---
payloads = []
truths = []
classes = list(data_templates.keys())

print(f"Generating {NUM_SAMPLES} samples...")

for _ in range(NUM_SAMPLES):
    # 1. Randomly choose a class for this row
    true_class = random.choice(classes)
    
    # 2. Randomly pick a corresponding text payload
    payload_text = random.choice(data_templates[true_class])
    
    # 3. Append to our lists
    payloads.append(payload_text)
    truths.append(true_class)

# 4. Create the Pandas DataFrame
df = pd.DataFrame({
    'payload': payloads,
    'truth': truths
})

# 5. Save the DataFrame to a CSV file
try:
    df.to_csv(FILENAME, index=False)
    print(f"\n✅ Successfully created '{FILENAME}' with {len(df)} rows.")
    print("Here is a preview of the data:")
    print(df.head())
except Exception as e:
    print(f"\n❌ Error saving file: {e}")