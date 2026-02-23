#!/bin/bash
# ROYALEY Frontend Fix: Hide voided predictions
# Run: cd /nvme0n1-disk/royaley && bash fix_frontend_void.sh

FILE="frontend/src/pages/Predictions/Predictions.tsx"

if [ ! -f "$FILE" ]; then
    echo "ERROR: $FILE not found. Run from project root."
    exit 1
fi

cp "$FILE" "${FILE}.bak"
echo "Backup: ${FILE}.bak"

python3 << 'PYEOF'
import sys

f = "frontend/src/pages/Predictions/Predictions.tsx"
content = open(f).read()
changes = 0

# Fix 1: Add 'void' to result type
old = "result: 'pending' | 'won' | 'lost' | 'push';"
new = "result: 'pending' | 'won' | 'lost' | 'push' | 'void';"
if old in content and new not in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  Fix 1: Added 'void' to result type")
elif new in content:
    print("  Fix 1: Already applied")

# Fix 2: Map void result from API
old = "else if (pred.result === 'push') result = 'push';"
new = "else if (pred.result === 'push') result = 'push';\n    else if (pred.result === 'void') result = 'void';"
if "pred.result === 'void'" not in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  Fix 2: Added void result mapping")
else:
    print("  Fix 2: Already applied")

# Fix 3: Filter void from display rows
old = "setRows(transformToFlatRows(data?.predictions || (Array.isArray(data) ? data : []), timezone, timeFormat));"
new = "setRows(transformToFlatRows(data?.predictions || (Array.isArray(data) ? data : []), timezone, timeFormat).filter(r => r.result !== 'void'));"
if old in content and new not in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  Fix 3: Filter void predictions from display")
elif new in content:
    print("  Fix 3: Already applied")

# Fix 4: Exclude void from graded stats
old = "const graded = statsRows.filter(r => r.result !== 'pending');"
new = "const graded = statsRows.filter(r => r.result !== 'pending' && r.result !== 'void');"
if old in content and new not in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  Fix 4: Exclude void from graded stats")
elif new in content:
    print("  Fix 4: Already applied")

# Fix 5: Exclude void from graded tab filter
old = "f = f.filter(r => r.result !== 'pending');"
new = "f = f.filter(r => r.result !== 'pending' && r.result !== 'void');"
if old in content and new not in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  Fix 5: Exclude void from graded tab")
elif new in content:
    print("  Fix 5: Already applied")

open(f, 'w').write(content)
print(f"\nApplied {changes} fixes")
PYEOF

echo ""
echo "Now rebuild frontend:"
echo "  docker compose up -d --build frontend"