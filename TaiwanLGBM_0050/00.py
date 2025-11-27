import ssl
import certifi

print("Certifi at:", certifi.where())
ctx = ssl.create_default_context(cafile=certifi.where())
print("SSL context created with certifi.")
print("SUCCESS!")
