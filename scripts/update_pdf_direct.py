import fitz
import os
import shutil

src_pdf = r"D:\Downloads\IPO_Listing_Gain_Predictor_Mini_Project_Report_BACKUP.pdf"
out_pdf = r"D:\Downloads\IPO_Listing_Gain_Predictor_Mini_Project_Report.pdf"
repo_pdf = r"d:\Pranav documents\IPOListingGain Predictor\reports\IPO_Listing_Gain_Predictor_Mini_Project_Report.pdf"

doc = fitz.open(src_pdf)

# ----------------------------------------------------
# PAGE 3 (Index 2)
# ----------------------------------------------------
page3 = doc[2]

# 1. Replace [N] with 35
rects_n = page3.search_for("[N]")
if rects_n:
    r = rects_n[0]
    # expand slightly to clear yellow highlight
    r_exp = fitz.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1)
    page3.add_redact_annot(r_exp, fill=(1, 1, 1))
    page3.apply_redactions()
    page3.insert_text((r.x0 + 1, r.y1 - 2.5), "35", fontname="tiro", fontsize=12.0, color=(0, 0, 0))

# 2. Replace yellow note on Page 3
note_text = "If any data is synthetic or sample data, write that here: it only demonstrates that the code works and must not be treated as real-market accuracy."
rects_note = page3.search_for(note_text)
if not rects_note:
    # try searching parts
    rects_note = page3.search_for("it only demonstrates that the code works")

# Let's inspect drawings / yellow rects on page 3
drawings_p3 = page3.get_drawings()
for d in drawings_p3:
    if d.get("fill") == (1.0, 1.0, 0.0) and d["rect"].y0 > 130 and d["rect"].y0 < 190:
        page3.add_redact_annot(d["rect"], fill=(1, 1, 1))
page3.apply_redactions()

# ----------------------------------------------------
# PAGE 4 (Index 3)
# ----------------------------------------------------
page4 = doc[3]

# 1. GitHub repo link
rects_gh = page4.search_for("[paste link]")
if rects_gh:
    r = rects_gh[0]
    page4.add_redact_annot(fitz.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2), fill=(1, 1, 1))
    page4.apply_redactions()
    # insert github link
    page4.insert_text(
        (r.x0, r.y1 - 2.5),
        "https://github.com/pranavkale1108/IPO-ListingGain-Predictor-",
        fontname="tiro",
        fontsize=10.0,
        color=(0, 0, 0.8)
    )

# 2. Streamlit link
rects_st = page4.search_for("[paste streamlit.app link]")
if rects_st:
    r = rects_st[0]
    page4.add_redact_annot(fitz.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2), fill=(1, 1, 1))
    page4.apply_redactions()
    page4.insert_text(
        (r.x0, r.y1 - 2.5),
        "http://localhost:8501",
        fontname="tiro",
        fontsize=10.0,
        color=(0, 0, 0.8)
    )

# ----------------------------------------------------
# PAGE 5 (Index 4) - RESULTS TABLE
# ----------------------------------------------------
page5 = doc[4]

# Remove the yellow highlight on the instruction sentence if present
drawings_p5 = page5.get_drawings()
for d in drawings_p5:
    if d.get("fill") == (1.0, 1.0, 0.0):
        # check if it's the sentence highlight or table
        if d["rect"].y0 < 110:
            page5.add_redact_annot(d["rect"], fill=(1, 1, 1))
page5.apply_redactions()
# Reinsert instruction sentence cleanly in black
instr_rects = page5.search_for("Copy the values from your")
# (Redaction above already wiped the text, let's see)

# Values to place in the Results table:
results = [
    "0.4611",           # Linear Regression (R2)
    "0.6283",           # Polynomial Regression (R2)
    "55.56%",           # Logistic Regression (Accuracy)
    "44.44%",           # SVM (Accuracy)
    "55.56%",           # k-NN (Accuracy)
    "44.44%"            # AdaBoost / Gradient Boosting (Accuracy)
]

rects_fill = page5.search_for("Fill after running")
print(f"Found {len(rects_fill)} instances of 'Fill after running'")

for idx, r in enumerate(rects_fill):
    val = results[idx] if idx < len(results) else "Done"
    # expand slightly to cover borders of yellow highlight
    r_exp = fitz.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 40, r.y1 + 2)
    page5.add_redact_annot(r_exp, fill=(1, 1, 1))

page5.apply_redactions()

# Now re-insert the actual values at the exact locations
for idx, r in enumerate(rects_fill):
    val = results[idx] if idx < len(results) else "Done"
    # y-baseline is roughly y1 - 2.5
    page5.insert_text(
        (r.x0 + 4, r.y1 - 2.5),
        val,
        fontname="tiro",
        fontsize=10.5,
        color=(0, 0, 0)
    )

doc.save(out_pdf)
shutil.copy2(out_pdf, repo_pdf)
print(f"Successfully updated PDF saved to {out_pdf} and {repo_pdf}")
