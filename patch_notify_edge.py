#!/usr/bin/env python3
"""
Patch scheduler_service.py to only send notifications for predictions with positive edge.
Prevents alerts for picks like -15.6% edge that have no real value.

Run from project root:
  python3 patch_notify_edge.py
"""

filepath = 'app/services/scheduling/scheduler_service.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

old = """WHERE p.created_at >= NOW() - INTERVAL '30 minutes'
                    ORDER BY p.confidence DESC"""

new = """WHERE p.created_at >= NOW() - INTERVAL '30 minutes'
                    AND p.edge > 0
                    ORDER BY p.confidence DESC"""

if old in content and 'AND p.edge > 0' not in content:
    content = content.replace(old, new)
    changes += 1
    print("✅ 1/1 Added edge > 0 filter to notification query")
else:
    print("⚠️  1/1 Already patched or target not found")

if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ Notification filter patched ({changes} changes)")
    print(f"  Verify: python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('OK')\"")
else:
    print("\n⚠️  No changes made")