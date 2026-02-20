import sys, json
sys.path.insert(0, r"C:/Users/krush/source/repos/lkml-reviews")
from build_reviews import build_review_html

with open(r"C:/Users/krush/source/repos/lkml-reviews/sample_data.json", encoding="utf-8") as f:
    data = json.load(f)

html = build_review_html(data)
out_path = r"C:/Users/krush/source/repos/lkml-reviews/sample_combined_review.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("Written OK, length:", len(html))