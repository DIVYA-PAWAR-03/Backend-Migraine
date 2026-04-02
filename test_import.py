import sys
import traceback

try:
    print("Importing app.routes.api...")
    from app.routes import api
    print(f"SUCCESS! Router has {len(api.router.routes)} routes")
    
    print("\nAll registered routes:")
    for i, route in enumerate(api.router.routes, 1):
        print(f"  {i}. {route.path} - {route.methods}")
        
except Exception as e:
    print(f"ERROR during import: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
