import requests
import json
import time

def test_streaming():
    url = "http://localhost:8000/ask/stream"
    payload = {"question": "What is NotchNet?"}
    headers = {"Content-Type": "application/json"}
    
    print(f"🚀 Testing streaming endpoint: {url}")
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as resp:
            if resp.status_code != 200:
                print(f"❌ Error: {resp.status_code} - {resp.text}")
                return
            
            print("✅ Connected. Receiving stream...")
            start_time = time.time()
            token_count = 0
            
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data == "[DONE]":
                            print("\n[DONE] Signal received.")
                            break
                        try:
                            json_data = json.loads(data)
                            if "answer" in json_data:
                                token = json_data["answer"]
                                print(token, end="", flush=True)
                                token_count += 1
                            elif "error" in json_data:
                                print(f"\n❌ Stream Error: {json_data['error']}")
                        except:
                            print(f"\n⚠️ Malformed data: {data}")
            
            duration = time.time() - start_time
            print(f"\n\n✅ Stream finished in {duration:.2f}s. Received {token_count} tokens.")
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")

if __name__ == "__main__":
    test_streaming()
