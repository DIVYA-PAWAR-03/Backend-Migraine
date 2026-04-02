from app.routes.api import router

print(f'Router has {len(router.routes)} routes registered')
print('\nRoutes:')
for route in router.routes:
    print(f'  {route.path} - {route.methods}')
