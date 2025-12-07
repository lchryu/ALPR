import requests

API_URL = "http://127.0.0.1:8000/alpr"
# IMAGE_PATH = "images/test.jpg"   # đổi path ở đây
IMAGE_PATH = "../data/test/images/test_xm4.jpg"   # đổi path ở đây

def test_alpr():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        res = requests.post(API_URL, files=files)

    print("\n=== API RESPONSE ===")
    try:
        print(res.json())
    except Exception:
        print("Response is not JSON:", res.text)


if __name__ == "__main__":
    test_alpr()
