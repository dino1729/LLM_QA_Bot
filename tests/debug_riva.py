
import sys
import importlib.util

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

package_name = "nvidia-riva-client"
spec = importlib.util.find_spec("riva.client")
if spec:
    print(f"riva.client found at: {spec.origin}")
else:
    print("riva.client NOT found")

try:
    import riva.client
    print("Successfully imported riva.client")
except ImportError as e:
    print(f"Failed to import riva.client: {e}")

try:
    import nvidia_riva_client
    print("Successfully imported nvidia_riva_client")
except ImportError as e:
    print(f"Failed to import nvidia_riva_client: {e}")
