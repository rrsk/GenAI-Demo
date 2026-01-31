from backend.lstm_service import get_lstm_service

# Test prediction
service = get_lstm_service()
forecast = service.predict_forecast('USER_00001')

print("[TEST] Forecast generated!")
print(f"Status: {forecast['status']}")
print(f"Model: {forecast.get('model', 'N/A')}")
print(f"Days forecasted: {len(forecast.get('predictions', []))}")

if forecast.get('predictions'):
    print(f"\nFirst day prediction:")
    print(f"  Date: {forecast['predictions'][0]['date']}")
    print(f"  Recovery: {forecast['predictions'][0]['recovery_score']}%")
    print(f"  Sleep: {forecast['predictions'][0]['sleep_hours']} hrs")
    print(f"  HRV: {forecast['predictions'][0]['hrv']}")
