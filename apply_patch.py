"""
بيدمج admin_patch.py جوه app.py أوتوماتيك.
الاستخدام (حط الـ 3 ملفات في نفس الفولدر):
    python apply_patch.py
بيعمل نسخة احتياطية app_backup.py وبعدين يعدّل app.py.
"""
import re
import shutil
import sys
from pathlib import Path

here = Path(__file__).parent
app_path, patch_path = here / "app.py", here / "admin_patch.py"
if not app_path.exists() or not patch_path.exists():
    sys.exit("لازم app.py و admin_patch.py يكونوا جنب apply_patch.py")

src = app_path.read_text(encoding="utf-8")
if "def admin_dashboard" in src:
    sys.exit("الباتش متطبق قبل كده على app.py.")

patch = patch_path.read_text(encoding="utf-8")
parts = re.split(r"^# =+\n# SECTION (\d)[^\n]*\n# =+\n", patch, flags=re.M)
# parts = [header, '1', body1, '2', body2, '3', body3]
sections = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
if set(sections) != {"1", "2", "3"}:
    sys.exit("admin_patch.py شكله اتعدّل: مش لاقي الأقسام 1 و2 و3.")

shutil.copy(app_path, here / "app_backup.py")

# 1) layout wide
src, n1 = re.subn(r'layout="centered"', 'layout="wide"', src, count=1)

# 2) استبدال auto_map
m = re.search(r"def auto_map\(columns, fields=None\):.*?(?=\ndef build_clean_df)", src, flags=re.S)
if not m:
    sys.exit("مش لاقي دالة auto_map في app.py.")
src = src[:m.start()] + sections["1"].strip("\n") + "\n\n" + src[m.end():]

# 3) استبدال قسم الأدمن من أول admin_upload_hunger لآخر الملف
i = src.find("def admin_upload_hunger():")
if i < 0:
    sys.exit("مش لاقي admin_upload_hunger في app.py.")
src = src[:i] + sections["2"].strip("\n") + "\n\n\n" + sections["3"].strip("\n") + "\n"

app_path.write_text(src, encoding="utf-8")
print("تم ✅  (layout wide:", bool(n1), ") — النسخة القديمة محفوظة في app_backup.py")
