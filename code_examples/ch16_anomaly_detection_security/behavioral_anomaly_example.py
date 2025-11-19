# Code from Chapter 16
# Book: Embeddings at Scale

"""
Behavioral Anomaly Detection

Architecture:
1. User behavior encoder: Actions → behavior embedding
2. Baseline behavior: Normal behavior for each user
3. Anomaly detector: Deviation from baseline
4. Adaptive learning: Update baseline with confirmed normal behavior

Use cases:
- Account takeover detection
- Insider threat detection
- Bot detection (automated account usage)
- Privilege escalation detection
"""

# Example: Account takeover detection
def behavioral_anomaly_example():
    """
    Account takeover detection for web application

    Normal behavior:
    - User logs in weekdays 9am-6pm from office
    - Accesses 10-20 pages per session
    - Session duration: 5-30 minutes

    Compromise indicators:
    - Login at unusual time (3am)
    - New location/device
    - Unusual actions (admin panel access, bulk data export)
    - High velocity (100+ pages in 5 minutes)
    """

    print("=== Account Takeover Detection ===")
    print("\nNormal baseline:")
    print("  Login time: Weekdays 9am-6pm")
    print("  Location: San Francisco office")
    print("  Device: MacBook Pro")
    print("  Actions: View dashboard, edit documents")
    print("  Velocity: 10-20 pages/session")

    print("\n--- Legitimate Session ---")
    print("Time: Tuesday 2pm")
    print("Location: San Francisco office")
    print("Device: MacBook Pro")
    print("Actions: View dashboard, edit report, send email")
    print("Velocity: 15 pages")
    print("→ Anomaly score: 0.05 (NORMAL)")

    print("\n--- Compromised Session ---")
    print("Time: Saturday 3am")
    print("Location: Unknown (Tor exit node)")
    print("Device: Windows PC (new)")
    print("Actions: Access admin panel, bulk export users, delete logs")
    print("Velocity: 150 pages")
    print("→ Anomaly score: 0.95 (ALERT: Possible account takeover)")

    print("\n--- Legitimate Travel ---")
    print("Time: Monday 10am")
    print("Location: New York office (business trip)")
    print("Device: MacBook Pro + iPhone")
    print("Actions: View dashboard, edit documents")
    print("Velocity: 12 pages")
    print("→ Anomaly score: 0.25 (MONITOR: New location, but normal actions)")

# Uncomment to run:
# behavioral_anomaly_example()
