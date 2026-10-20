import re
s = open("app.py", encoding="utf-8").read()
pattern = r'    if top\[1\] in \("High", "Critical"\):\n.*?st\.success\("No overcrowding alert"\)'
new = '    if top[1] in ("High", "Critical"):\n        st.error("OVERCROWDED")\n    else:\n        st.success("NOT OVERCROWDED")'
s2, n = re.subn(pattern, new, s, flags=re.S)
open("app.py", "w", encoding="utf-8").write(s2)
print("replaced blocks:", n)
