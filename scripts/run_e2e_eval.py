"""End-to-end evaluation of the clinical notes RAG pipeline.

Generates synthetic clinical notes, builds a FAISS index with real embeddings,
runs retrieval queries, and measures recall@k. Tests the NLI grounding checker
on known-grounded and known-ungrounded answer pairs.

All metrics are measured, not fabricated.
"""
from __future__ import annotations

import json
import random
import sys
import time

import numpy as np
from langchain_core.documents import Document

sys.path.insert(0, "src")

from clinical_rag.config import RAGConfig
from clinical_rag.grounding import GroundingChecker
from clinical_rag.index import FAISSIndex
from clinical_rag.ingest.chunker import chunk_documents

# ---------------------------------------------------------------------------
# 1. Generate synthetic clinical notes
# ---------------------------------------------------------------------------

SPECIALTIES = [
    "Orthopedics", "Endocrinology", "Pulmonology", "Cardiology",
    "Gastroenterology", "Neurology", "Dermatology", "Oncology",
    "Nephrology", "Rheumatology", "Psychiatry", "Ophthalmology",
    "Urology", "Obstetrics", "Pediatrics", "Infectious Disease",
    "Hematology", "ENT", "General Surgery", "Emergency Medicine",
]

NOTE_TEMPLATES: list[dict] = [
    {
        "condition": "rotator cuff tear",
        "specialty": "Orthopedics",
        "hpi": "Patient is a {age}-year-old {sex} presenting with right shoulder pain for {duration} weeks. Pain worsens with overhead activities. Reports difficulty sleeping on affected side. Denies numbness or tingling in the arm.",
        "exam": "Right shoulder: tenderness over greater tuberosity. Positive Neer impingement sign. Positive empty can test. Active forward flexion limited to {flexion} degrees. External rotation strength {er_strength}/5.",
        "assessment": "Right rotator cuff tear, likely {tear_type}. MRI recommended to determine extent of tear and guide treatment planning.",
        "plan": "1. MRI right shoulder without contrast\n2. NSAIDs for pain management (ibuprofen 600mg TID)\n3. Physical therapy referral for rotator cuff strengthening\n4. Follow-up in {followup} weeks with MRI results\n5. Discuss surgical options if conservative management fails",
    },
    {
        "condition": "type 2 diabetes mellitus",
        "specialty": "Endocrinology",
        "hpi": "Patient is a {age}-year-old {sex} with type 2 diabetes mellitus diagnosed {years_ago} years ago. Current HbA1c is {a1c}%. Reports polyuria and polydipsia. Checks blood glucose {bg_freq}. Home readings range {bg_low}-{bg_high} mg/dL fasting.",
        "exam": "Vitals: BP {bp}, HR {hr}, BMI {bmi}. Foot exam: {foot_findings}. Monofilament testing {monofilament}. Fundoscopic exam deferred to ophthalmology.",
        "assessment": "Type 2 diabetes mellitus, {control}. HbA1c {a1c}% indicates {a1c_interp}.",
        "plan": "1. {med_change}\n2. Continue metformin {met_dose}mg BID\n3. Diabetic diet counseling\n4. Ophthalmology referral for dilated eye exam\n5. Repeat HbA1c in 3 months\n6. Fasting lipid panel and comprehensive metabolic panel",
    },
    {
        "condition": "community-acquired pneumonia",
        "specialty": "Pulmonology",
        "hpi": "Patient is a {age}-year-old {sex} presenting with {duration}-day history of productive cough with {sputum_color} sputum, fever to {temp}F, and shortness of breath. Reports chills and malaise. Denies recent travel or sick contacts.",
        "exam": "Vitals: Temp {temp}F, HR {hr}, RR {rr}, SpO2 {spo2}% on room air. Lung exam: {lung_findings}. No accessory muscle use.",
        "assessment": "Community-acquired pneumonia, {severity}. CURB-65 score {curb65}.",
        "plan": "1. Chest X-ray PA and lateral\n2. {abx_regimen}\n3. Supportive care: fluids, rest, acetaminophen PRN for fever\n4. Sputum culture if not improving\n5. Follow-up chest X-ray in {followup} weeks\n6. Return precautions for worsening dyspnea or hemoptysis",
    },
    {
        "condition": "atrial fibrillation",
        "specialty": "Cardiology",
        "hpi": "Patient is a {age}-year-old {sex} presenting with palpitations for {duration} days. Reports intermittent rapid heartbeat and mild dyspnea on exertion. Denies chest pain, syncope, or lower extremity edema. Past medical history notable for hypertension.",
        "exam": "Vitals: BP {bp}, HR {hr} and irregularly irregular, RR 16. Cardiac exam: irregular rhythm, no murmurs, rubs, or gallops. JVP not elevated. Lungs clear. No peripheral edema.",
        "assessment": "New-onset atrial fibrillation. CHA2DS2-VASc score {chads}. HAS-BLED score {hasbled}.",
        "plan": "1. ECG to confirm atrial fibrillation\n2. Echocardiogram to evaluate structure and function\n3. TSH to rule out thyroid disease\n4. {rate_control}\n5. {anticoag}\n6. Cardiology follow-up in {followup} weeks",
    },
    {
        "condition": "gastroesophageal reflux disease",
        "specialty": "Gastroenterology",
        "hpi": "Patient is a {age}-year-old {sex} with {duration}-month history of heartburn and regurgitation. Symptoms worse after meals and when lying flat. Reports occasional dysphagia with solid foods. Denies weight loss, melena, or hematemesis.",
        "exam": "Vitals stable. Abdomen soft, nontender, no masses. No epigastric tenderness. Oropharynx clear without erythema.",
        "assessment": "Gastroesophageal reflux disease (GERD), {severity}. Alarm features: {alarm}.",
        "plan": "1. PPI therapy: {ppi} {ppi_dose}mg daily for 8 weeks\n2. Lifestyle modifications: elevate head of bed, avoid eating 3 hours before bedtime, limit caffeine and alcohol\n3. {additional}\n4. Follow-up in 8 weeks to reassess symptoms",
    },
    {
        "condition": "major depressive disorder",
        "specialty": "Psychiatry",
        "hpi": "Patient is a {age}-year-old {sex} presenting with depressed mood for {duration} weeks. Reports anhedonia, insomnia with early morning awakening, poor concentration, fatigue, and feelings of worthlessness. Appetite {appetite}. Denies suicidal ideation, plan, or intent.",
        "exam": "Mental status: Appearance well-groomed. Psychomotor {psychomotor}. Mood 'down.' Affect constricted. Thought process linear and goal-directed. No hallucinations or delusions. Insight and judgment fair.",
        "assessment": "Major depressive disorder, {episode}, {severity}. PHQ-9 score: {phq9}.",
        "plan": "1. Start {ssri} {ssri_dose}mg daily\n2. Referral to psychotherapy (CBT)\n3. Sleep hygiene counseling\n4. Safety plan reviewed, emergency contacts confirmed\n5. Follow-up in {followup} weeks to assess medication response\n6. PHQ-9 to be repeated at follow-up",
    },
    {
        "condition": "chronic kidney disease",
        "specialty": "Nephrology",
        "hpi": "Patient is a {age}-year-old {sex} with chronic kidney disease stage {ckd_stage}, eGFR {egfr} mL/min. History of hypertension and diabetes. Presents for routine follow-up. Reports mild fatigue. Denies edema, hematuria, or changes in urine output.",
        "exam": "Vitals: BP {bp}, HR {hr}. No peripheral edema. Lungs clear. Abdomen soft, no CVA tenderness.",
        "assessment": "CKD stage {ckd_stage}, stable. eGFR {egfr}. Etiology likely diabetic nephropathy.",
        "plan": "1. Continue ACE inhibitor for renoprotection\n2. Blood pressure goal <130/80\n3. Low sodium diet, protein restriction to {protein}g/kg/day\n4. Repeat BMP and urine albumin-to-creatinine ratio in 3 months\n5. Avoid nephrotoxins (NSAIDs, contrast)\n6. Nephrology follow-up in {followup} months",
    },
    {
        "condition": "migraine headache",
        "specialty": "Neurology",
        "hpi": "Patient is a {age}-year-old {sex} with recurrent headaches occurring {freq} times per month. Headaches are {location}, pulsating in quality, moderate to severe intensity, lasting {duration} hours. Associated with {assoc}. Triggers include {triggers}.",
        "exam": "Neurological exam: Cranial nerves II-XII intact. Motor strength 5/5 throughout. Sensation intact. Reflexes 2+ symmetric. No papilledema. No nuchal rigidity.",
        "assessment": "Migraine {with_aura}, {frequency}. Meeting criteria for preventive therapy.",
        "plan": "1. Abortive: {abortive}\n2. Preventive: {preventive}\n3. Headache diary to track frequency and triggers\n4. Lifestyle modifications: regular sleep, exercise, hydration\n5. MRI brain if no prior imaging\n6. Follow-up in {followup} weeks",
    },
    {
        "condition": "osteoarthritis of the knee",
        "specialty": "Orthopedics",
        "hpi": "Patient is a {age}-year-old {sex} with {duration}-month history of bilateral knee pain, worse on the {side}. Pain aggravated by walking and stair climbing, relieved by rest. Reports morning stiffness lasting {stiffness} minutes. No history of trauma.",
        "exam": "Bilateral knees: {side} with mild effusion, tenderness over medial joint line. Crepitus with range of motion. Flexion {flexion} degrees. No ligamentous laxity. McMurray test negative.",
        "assessment": "Osteoarthritis of the {side} knee, Kellgren-Lawrence grade {kl_grade}.",
        "plan": "1. Weight-bearing X-rays of bilateral knees\n2. Acetaminophen 1000mg TID as first-line\n3. Topical NSAIDs (diclofenac gel)\n4. Physical therapy for quadriceps strengthening\n5. Weight loss counseling (current BMI {bmi})\n6. Consider intra-articular corticosteroid injection if conservative measures fail",
    },
    {
        "condition": "hypothyroidism",
        "specialty": "Endocrinology",
        "hpi": "Patient is a {age}-year-old {sex} with hypothyroidism on levothyroxine {levo_dose}mcg daily. Presents for routine follow-up. Reports {symptoms}. Weight {weight_change} {weight_amt} lbs over past {weight_period} months.",
        "exam": "Vitals: BP {bp}, HR {hr}. Thyroid: {thyroid_exam}. Skin: {skin_findings}. Reflexes: {reflexes}.",
        "assessment": "Hypothyroidism, {control}. TSH {tsh} mIU/L (goal 0.4-4.0).",
        "plan": "1. {dose_change}\n2. Continue taking levothyroxine on empty stomach, 30 minutes before food\n3. Repeat TSH in {followup} weeks\n4. Lipid panel to monitor cardiovascular risk\n5. Bone density screening if indicated",
    },
]


def _rand_choice(*options):
    return random.choice(options)


def generate_note(template: dict, note_id: int) -> Document:
    age = random.randint(25, 85)
    sex = _rand_choice("male", "female")
    params: dict = {
        "age": age, "sex": sex,
        "duration": random.randint(1, 24),
        "followup": random.randint(2, 8),
        "bp": f"{random.randint(110,160)}/{random.randint(65,95)}",
        "hr": random.randint(60, 110),
        "bmi": round(random.uniform(22, 40), 1),
        # Rotator cuff
        "flexion": random.randint(90, 160),
        "er_strength": random.randint(3, 5),
        "tear_type": _rand_choice("partial-thickness", "full-thickness"),
        # Diabetes
        "years_ago": random.randint(1, 20),
        "a1c": round(random.uniform(6.5, 11.0), 1),
        "bg_freq": _rand_choice("daily", "twice daily", "3 times daily"),
        "bg_low": random.randint(90, 160),
        "bg_high": random.randint(180, 320),
        "foot_findings": _rand_choice("intact sensation, no ulcers", "decreased sensation left great toe", "callus formation bilateral"),
        "monofilament": _rand_choice("intact bilaterally", "decreased right foot", "intact"),
        "control": _rand_choice("poorly controlled", "suboptimally controlled", "well controlled"),
        "a1c_interp": _rand_choice("need for therapy adjustment", "adequate control", "poor glycemic control"),
        "med_change": _rand_choice("Add glipizide 5mg daily", "Increase metformin to 1000mg BID", "Add empagliflozin 10mg daily", "Add semaglutide 0.25mg weekly"),
        "met_dose": _rand_choice(500, 850, 1000),
        # Pneumonia
        "sputum_color": _rand_choice("yellow-green", "rust-colored", "white"),
        "temp": round(random.uniform(100.4, 103.5), 1),
        "rr": random.randint(18, 28),
        "spo2": random.randint(89, 97),
        "lung_findings": _rand_choice("crackles in right lower lobe", "bilateral basilar crackles", "decreased breath sounds left base with dullness to percussion"),
        "severity": _rand_choice("mild", "moderate", "severe"),
        "curb65": random.randint(0, 3),
        "abx_regimen": _rand_choice("Azithromycin 500mg day 1 then 250mg days 2-5", "Amoxicillin-clavulanate 875mg BID x 7 days", "Levofloxacin 750mg daily x 5 days"),
        # AFib
        "chads": random.randint(1, 5),
        "hasbled": random.randint(0, 3),
        "rate_control": _rand_choice("Metoprolol succinate 25mg daily", "Diltiazem 120mg daily"),
        "anticoag": _rand_choice("Start apixaban 5mg BID", "Start rivaroxaban 20mg daily with food", "Discuss anticoagulation risks and benefits"),
        # GERD
        "alarm": _rand_choice("absent", "dysphagia present, EGD recommended"),
        "ppi": _rand_choice("omeprazole", "pantoprazole", "esomeprazole"),
        "ppi_dose": _rand_choice(20, 40),
        "additional": _rand_choice("EGD if symptoms persist after PPI trial", "H. pylori testing", "No additional workup at this time"),
        # Psych
        "appetite": _rand_choice("decreased with 5lb weight loss", "increased with carbohydrate craving", "unchanged"),
        "psychomotor": _rand_choice("retardation", "normal", "mild agitation"),
        "episode": _rand_choice("single episode", "recurrent"),
        "phq9": random.randint(10, 24),
        "ssri": _rand_choice("sertraline", "escitalopram", "fluoxetine"),
        "ssri_dose": _rand_choice(25, 50, 10, 20),
        # CKD
        "ckd_stage": _rand_choice("3a", "3b", "4"),
        "egfr": random.randint(15, 55),
        "protein": round(random.uniform(0.6, 0.8), 1),
        # Migraine
        "freq": random.randint(4, 15),
        "location": _rand_choice("unilateral, right temporal", "bilateral frontal", "left-sided"),
        "assoc": _rand_choice("nausea and photophobia", "nausea, photophobia, and phonophobia", "nausea only"),
        "triggers": _rand_choice("stress, poor sleep, and red wine", "bright lights and caffeine withdrawal", "menstruation and weather changes"),
        "with_aura": _rand_choice("with aura", "without aura"),
        "frequency": _rand_choice("episodic", "chronic (>15 days/month)"),
        "abortive": _rand_choice("sumatriptan 100mg PO PRN", "rizatriptan 10mg ODT PRN"),
        "preventive": _rand_choice("topiramate 25mg daily, titrate to 100mg", "propranolol 40mg BID", "amitriptyline 25mg at bedtime"),
        # OA
        "side": _rand_choice("right", "left"),
        "stiffness": random.randint(10, 45),
        "kl_grade": random.randint(2, 4),
        # Hypothyroidism
        "levo_dose": _rand_choice(25, 50, 75, 100, 125),
        "symptoms": _rand_choice("feeling well, no complaints", "mild fatigue and cold intolerance", "constipation and dry skin"),
        "weight_change": _rand_choice("gained", "lost", "stable at"),
        "weight_amt": random.randint(2, 10),
        "weight_period": random.randint(2, 6),
        "thyroid_exam": _rand_choice("no enlargement, no nodules", "mildly enlarged, no nodules", "small firm nodule right lobe"),
        "skin_findings": _rand_choice("normal", "dry skin", "mild pretibial myxedema"),
        "reflexes": _rand_choice("2+ symmetric", "1+ with delayed relaxation phase", "normal"),
        "tsh": round(random.uniform(0.1, 12.0), 2),
        "dose_change": _rand_choice("Increase levothyroxine to next dose level", "Continue current dose", "Decrease levothyroxine by 25mcg"),
    }

    note_text = f"""CHIEF COMPLAINT
{template['condition'].title()}

HISTORY OF PRESENT ILLNESS
{template['hpi'].format(**params)}

PHYSICAL EXAMINATION
{template['exam'].format(**params)}

ASSESSMENT AND PLAN
{template['assessment'].format(**params)}

PLAN
{template['plan'].format(**params)}"""

    return Document(
        page_content=note_text,
        metadata={
            "note_id": f"NOTE-{note_id:04d}",
            "specialty": template["specialty"],
            "condition": template["condition"],
        },
    )


def generate_notes(n: int = 220) -> list[Document]:
    random.seed(42)
    notes = []
    for i in range(n):
        template = NOTE_TEMPLATES[i % len(NOTE_TEMPLATES)]
        notes.append(generate_note(template, i))
    return notes


# ---------------------------------------------------------------------------
# 2. Retrieval queries with known relevant note IDs
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    {"query": "What are the treatment options for rotator cuff tear?", "condition": "rotator cuff tear"},
    {"query": "How is type 2 diabetes managed with medications?", "condition": "type 2 diabetes mellitus"},
    {"query": "What antibiotics are used for community-acquired pneumonia?", "condition": "community-acquired pneumonia"},
    {"query": "What is the workup for new-onset atrial fibrillation?", "condition": "atrial fibrillation"},
    {"query": "How is GERD treated with proton pump inhibitors?", "condition": "gastroesophageal reflux disease"},
    {"query": "What medications are prescribed for major depressive disorder?", "condition": "major depressive disorder"},
    {"query": "How is chronic kidney disease monitored?", "condition": "chronic kidney disease"},
    {"query": "What preventive medications are used for migraine headaches?", "condition": "migraine headache"},
    {"query": "What is the management of knee osteoarthritis?", "condition": "osteoarthritis of the knee"},
    {"query": "How is hypothyroidism monitored and treated?", "condition": "hypothyroidism"},
    {"query": "What physical exam findings suggest a rotator cuff injury?", "condition": "rotator cuff tear"},
    {"query": "What is the HbA1c target for diabetic patients?", "condition": "type 2 diabetes mellitus"},
    {"query": "When should pneumonia patients be hospitalized?", "condition": "community-acquired pneumonia"},
    {"query": "What is the CHA2DS2-VASc score used for?", "condition": "atrial fibrillation"},
    {"query": "What lifestyle modifications help with acid reflux?", "condition": "gastroesophageal reflux disease"},
    {"query": "What is the PHQ-9 questionnaire used for?", "condition": "major depressive disorder"},
    {"query": "What dietary restrictions apply to chronic kidney disease?", "condition": "chronic kidney disease"},
    {"query": "What are common migraine triggers?", "condition": "migraine headache"},
    {"query": "When is knee replacement surgery considered?", "condition": "osteoarthritis of the knee"},
    {"query": "What are the symptoms of hypothyroidism?", "condition": "hypothyroidism"},
    {"query": "What imaging is ordered for shoulder pain?", "condition": "rotator cuff tear"},
    {"query": "What is metformin used for in diabetes treatment?", "condition": "type 2 diabetes mellitus"},
    {"query": "What sputum characteristics are seen in pneumonia?", "condition": "community-acquired pneumonia"},
    {"query": "What rate control medications are used for atrial fibrillation?", "condition": "atrial fibrillation"},
    {"query": "What alarm features warrant endoscopy in GERD?", "condition": "gastroesophageal reflux disease"},
    {"query": "What is cognitive behavioral therapy used for in depression?", "condition": "major depressive disorder"},
    {"query": "Why should NSAIDs be avoided in kidney disease?", "condition": "chronic kidney disease"},
    {"query": "What is sumatriptan used for?", "condition": "migraine headache"},
    {"query": "What physical therapy exercises help with knee arthritis?", "condition": "osteoarthritis of the knee"},
    {"query": "How often should TSH be monitored in hypothyroidism?", "condition": "hypothyroidism"},
    {"query": "What is the Neer impingement test?", "condition": "rotator cuff tear"},
    {"query": "What oral medications treat type 2 diabetes?", "condition": "type 2 diabetes mellitus"},
    {"query": "What oxygen saturation level is concerning in pneumonia?", "condition": "community-acquired pneumonia"},
    {"query": "What anticoagulant options exist for atrial fibrillation?", "condition": "atrial fibrillation"},
    {"query": "How long should PPI therapy last for GERD?", "condition": "gastroesophageal reflux disease"},
    {"query": "What SSRI medications treat depression?", "condition": "major depressive disorder"},
    {"query": "What blood pressure target applies in chronic kidney disease?", "condition": "chronic kidney disease"},
    {"query": "How are chronic migraines defined?", "condition": "migraine headache"},
    {"query": "What is the Kellgren-Lawrence grading system?", "condition": "osteoarthritis of the knee"},
    {"query": "What levothyroxine doses are commonly prescribed?", "condition": "hypothyroidism"},
]


# ---------------------------------------------------------------------------
# 3. Grounding checker test pairs
# ---------------------------------------------------------------------------

GROUNDING_TESTS = [
    {
        "context": "Patient has a right rotator cuff tear. MRI recommended. Physical therapy referral for rotator cuff strengthening. NSAIDs for pain management. Follow-up in 4 weeks with MRI results.",
        "answer_grounded": "The patient has a right rotator cuff tear. An MRI was recommended and the patient was referred to physical therapy. NSAIDs were prescribed for pain.",
        "answer_ungrounded": "The patient had successful rotator cuff surgery with full recovery. They were cleared to return to competitive sports after 6 weeks.",
    },
    {
        "context": "Type 2 diabetes mellitus, poorly controlled. HbA1c 9.2%. Continue metformin 1000mg BID. Add empagliflozin 10mg daily. Diabetic diet counseling. Repeat HbA1c in 3 months.",
        "answer_grounded": "The patient's diabetes is poorly controlled with an HbA1c of 9.2%. Empagliflozin was added to the existing metformin regimen.",
        "answer_ungrounded": "The patient's diabetes is in remission after bariatric surgery. All medications were discontinued and HbA1c normalized to 5.1%.",
    },
    {
        "context": "Community-acquired pneumonia, moderate severity. Crackles in right lower lobe. SpO2 94% on room air. CURB-65 score 2. Started on levofloxacin 750mg daily for 5 days.",
        "answer_grounded": "The patient was diagnosed with moderate community-acquired pneumonia with crackles in the right lower lobe. Levofloxacin was prescribed for 5 days.",
        "answer_ungrounded": "The patient was diagnosed with tuberculosis after positive AFB smear. Initiated on RIPE therapy with rifampin, isoniazid, pyrazinamide, and ethambutol.",
    },
    {
        "context": "New-onset atrial fibrillation. CHA2DS2-VASc score 3. Started apixaban 5mg BID. Metoprolol succinate 25mg daily for rate control. ECG and echocardiogram ordered.",
        "answer_grounded": "The patient was started on apixaban for anticoagulation given a CHA2DS2-VASc score of 3. Metoprolol was prescribed for rate control.",
        "answer_ungrounded": "The patient underwent successful catheter ablation and was cured of atrial fibrillation. No further anticoagulation needed.",
    },
    {
        "context": "Major depressive disorder, recurrent, moderate. PHQ-9 score 16. Started sertraline 50mg daily. Referral to psychotherapy. Follow-up in 4 weeks.",
        "answer_grounded": "The patient has recurrent moderate depression with a PHQ-9 of 16. Sertraline was started and psychotherapy was recommended.",
        "answer_ungrounded": "The patient was hospitalized for psychotic depression with auditory hallucinations. Started on olanzapine and electroconvulsive therapy.",
    },
    {
        "context": "CKD stage 3b, eGFR 38. Continue ACE inhibitor. Low sodium diet, protein restriction to 0.7g/kg/day. Avoid nephrotoxins. Nephrology follow-up in 3 months.",
        "answer_grounded": "The patient has CKD stage 3b with an eGFR of 38. Dietary restrictions include low sodium and protein limitation to 0.7g/kg/day. NSAIDs should be avoided.",
        "answer_ungrounded": "The patient progressed to end-stage renal disease and was initiated on hemodialysis three times per week via AV fistula.",
    },
    {
        "context": "Migraine without aura, episodic. Sumatriptan 100mg PO PRN for acute attacks. Starting topiramate 25mg daily for prevention, titrate to 100mg. Headache diary recommended.",
        "answer_grounded": "The patient has episodic migraine without aura. Sumatriptan was prescribed for acute episodes and topiramate was started for prevention.",
        "answer_ungrounded": "The patient was diagnosed with a brain tumor causing secondary headaches. Urgent neurosurgery consultation was placed.",
    },
    {
        "context": "Osteoarthritis right knee, Kellgren-Lawrence grade 3. Acetaminophen as first-line. Physical therapy for quadriceps strengthening. Consider corticosteroid injection if conservative measures fail.",
        "answer_grounded": "The patient has grade 3 osteoarthritis of the right knee. Initial treatment includes acetaminophen and physical therapy. Corticosteroid injection is an option if needed.",
        "answer_ungrounded": "The patient underwent total knee arthroplasty with excellent outcomes. Full weight bearing was achieved at 4 weeks post-operatively.",
    },
]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CLINICAL NOTES RAG - END-TO-END EVALUATION")
    print("=" * 70)

    # --- Generate notes ---
    print("\n[1/5] Generating synthetic clinical notes...")
    notes = generate_notes(220)
    print(f"  Generated {len(notes)} notes across {len(set(n.metadata['condition'] for n in notes))} conditions")

    # --- Chunk ---
    print("\n[2/5] Chunking clinical notes...")
    t0 = time.time()
    chunks = chunk_documents(notes, chunk_size=512, chunk_overlap=64)
    chunk_time = time.time() - t0
    print(f"  {len(chunks)} chunks from {len(notes)} notes ({chunk_time:.2f}s)")
    sections_found = set()
    for c in chunks:
        sections_found.add(c.metadata.get("section", "UNKNOWN"))
    print(f"  Sections detected: {sorted(sections_found)}")

    # --- Build FAISS index with real embeddings ---
    print("\n[3/5] Building FAISS index with sentence-transformers...")
    config = RAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
    )

    t0 = time.time()
    index = FAISSIndex.build(chunks, config)
    index_time = time.time() - t0
    print(f"  Indexed {index.index.ntotal} vectors in {index_time:.2f}s")

    # --- Retrieval evaluation ---
    print("\n[4/5] Running retrieval evaluation ({} queries)...".format(len(EVAL_QUERIES)))

    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    retrieval_latencies = []

    for eq in EVAL_QUERIES:
        query = eq["query"]
        target_condition = eq["condition"]

        t0 = time.time()
        results = index.search(query, top_k=20)
        latency = time.time() - t0
        retrieval_latencies.append(latency)

        retrieved_conditions_5 = [doc.metadata.get("condition") for doc, _ in results[:5]]
        retrieved_conditions_10 = [doc.metadata.get("condition") for doc, _ in results[:10]]
        retrieved_conditions_20 = [doc.metadata.get("condition") for doc, _ in results[:20]]

        recall_at_5.append(1.0 if target_condition in retrieved_conditions_5 else 0.0)
        recall_at_10.append(1.0 if target_condition in retrieved_conditions_10 else 0.0)
        recall_at_20.append(1.0 if target_condition in retrieved_conditions_20 else 0.0)

    mean_recall_5 = np.mean(recall_at_5)
    mean_recall_10 = np.mean(recall_at_10)
    mean_recall_20 = np.mean(recall_at_20)
    mean_latency = np.mean(retrieval_latencies) * 1000
    p95_latency = np.percentile(retrieval_latencies, 95) * 1000

    print(f"\n  Retrieval Results:")
    print(f"    Recall@5:   {mean_recall_5:.1%} ({int(sum(recall_at_5))}/{len(recall_at_5)} queries)")
    print(f"    Recall@10:  {mean_recall_10:.1%} ({int(sum(recall_at_10))}/{len(recall_at_10)} queries)")
    print(f"    Recall@20:  {mean_recall_20:.1%} ({int(sum(recall_at_20))}/{len(recall_at_20)} queries)")
    print(f"    Mean latency: {mean_latency:.1f}ms")
    print(f"    P95 latency:  {p95_latency:.1f}ms")

    # --- NLI Grounding evaluation ---
    print("\n[5/5] Running NLI grounding evaluation ({} test pairs)...".format(len(GROUNDING_TESTS)))

    checker = GroundingChecker(config)

    grounded_scores = []
    ungrounded_scores = []
    grounded_correct = 0
    ungrounded_correct = 0

    for test in GROUNDING_TESTS:
        ctx_docs = [Document(page_content=test["context"])]

        g_result = checker.check(test["answer_grounded"], ctx_docs)
        grounded_scores.append(g_result.score)
        if g_result.is_grounded:
            grounded_correct += 1

        u_result = checker.check(test["answer_ungrounded"], ctx_docs)
        ungrounded_scores.append(u_result.score)
        if not u_result.is_grounded:
            ungrounded_correct += 1

    n_tests = len(GROUNDING_TESTS)
    grounding_accuracy = (grounded_correct + ungrounded_correct) / (2 * n_tests)
    grounded_detection_rate = grounded_correct / n_tests
    ungrounded_rejection_rate = ungrounded_correct / n_tests

    print(f"\n  NLI Grounding Results (nli-deberta-v3-base):")
    print(f"    Grounded answers correctly accepted:   {grounded_correct}/{n_tests} ({grounded_detection_rate:.1%})")
    print(f"    Ungrounded answers correctly rejected:  {ungrounded_correct}/{n_tests} ({ungrounded_rejection_rate:.1%})")
    print(f"    Overall grounding accuracy:             {grounding_accuracy:.1%}")
    print(f"    Mean entailment score (grounded):       {np.mean(grounded_scores):.3f}")
    print(f"    Mean entailment score (ungrounded):     {np.mean(ungrounded_scores):.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY OF MEASURED METRICS")
    print("=" * 70)
    summary = {
        "notes_generated": len(notes),
        "chunks_created": len(chunks),
        "vectors_indexed": index.index.ntotal,
        "embedding_model": config.embedding_model,
        "grounding_model": config.grounding_model,
        "queries_evaluated": len(EVAL_QUERIES),
        "recall_at_5": round(mean_recall_5, 3),
        "recall_at_10": round(mean_recall_10, 3),
        "recall_at_20": round(mean_recall_20, 3),
        "mean_retrieval_latency_ms": round(mean_latency, 1),
        "p95_retrieval_latency_ms": round(p95_latency, 1),
        "grounding_tests": n_tests,
        "grounded_detection_rate": round(grounded_detection_rate, 3),
        "ungrounded_rejection_rate": round(ungrounded_rejection_rate, 3),
        "grounding_accuracy": round(grounding_accuracy, 3),
        "mean_entailment_grounded": round(float(np.mean(grounded_scores)), 3),
        "mean_entailment_ungrounded": round(float(np.mean(ungrounded_scores)), 3),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save results to file
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/e2e_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to results/e2e_eval_results.json")

    return summary


if __name__ == "__main__":
    summary = main()
