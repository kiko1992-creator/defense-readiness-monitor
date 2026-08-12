"""One-off check: what indicator codes actually exist in the Worldwide
Governance Indicators database (source id 3), so we stop guessing."""

import wbgapi as wb

wb.db = 3
for row in wb.series.list(q="stability"):
    print(row)
print("---")
for row in wb.series.list(q="rule of law"):
    print(row)