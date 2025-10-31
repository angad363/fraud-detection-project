# test_api.py - Test the fraud detection API

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health check"""
    print("="*60)
    print("TEST 1: Health Check")
    print("="*60)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction():
    """Test single transaction prediction"""
    print("="*60)
    print("TEST 2: Single Transaction Prediction")
    print("="*60)

    # Sample transaction (this is a legitimate transaction)
    transaction = {
        "Time": 406,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }

    response = requests.post(f"{API_URL}/predict", json=transaction)
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_prediction():
    """Test batch predictions"""
    print("="*60)
    print("TEST 3: Batch Predictions")
    print("="*60)

    # Multiple transactions
    transactions = [
        {
            "Time": 406,
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        },
        {
            "Time": 500,
            "V1": 2.5, "V2": 1.8, "V3": -3.2, "V4": 2.1,
            "V5": 1.5, "V6": -2.3, "V7": 1.9, "V8": -1.2,
            "V9": 0.8, "V10": 1.4, "V11": -2.1, "V12": 1.7,
            "V13": -1.8, "V14": 2.5, "V15": -1.3, "V16": 1.1,
            "V17": -0.9, "V18": 1.6, "V19": -1.4, "V20": 0.7,
            "V21": 1.2, "V22": -0.8, "V23": 1.5, "V24": -1.1,
            "V25": 0.9, "V26": 1.3, "V27": -1.0, "V28": 0.6,
            "Amount": 2500.00  # Suspiciously high amount
        }
    ]

    response = requests.post(f"{API_URL}/predict_batch", json=transactions)
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_performance():
    """Test API response time"""
    print("="*60)
    print("TEST 4: Performance Test")
    print("="*60)

    import time

    transaction = {
        "Time": 406,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62
    }

    n_requests = 100
    print(f"Sending {n_requests} requests...")

    times = []
    for i in range(n_requests):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=transaction)
        end = time.time()
        times.append(end - start)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_requests} requests")

    print(f"\nPerformance Results:")
    print(f"  Total requests: {n_requests}")
    print(f"  Average response time: {np.mean(times)*1000:.2f} ms")
    print(f"  Min response time: {np.min(times)*1000:.2f} ms")
    print(f"  Max response time: {np.max(times)*1000:.2f} ms")
    print(f"  95th percentile: {np.percentile(times, 95)*1000:.2f} ms")
    print()

if __name__ == "__main__":
    import numpy as np

    print("\n" + "="*60)
    print("FRAUD DETECTION API - COMPREHENSIVE TESTS")
    print("="*60 + "\n")

    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_performance()

        print("="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API")
        print("   Make sure the API is running: python app.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")