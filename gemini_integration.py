"""
Integration with Google's Gemini API for generating clinical narratives,
explanations, and patient-friendly summaries.
Automatically selects an available model and incorporates dataset examples.
"""

import os
import warnings
# Suppress FutureWarning regarding google.generativeai deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import google.generativeai as genai
from dotenv import load_dotenv
import config


load_dotenv()

_env_keys = [
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GENAI_API_KEY",
]

def _resolve_key():
    key = getattr(config, "GEMINI_API_KEY", None)
    
    if not key:
        for name in _env_keys:
            envv = os.getenv(name)
            if envv:
                key = envv
                break
    
    if not key:
        return None
    if key == "YOUR_GEMINI_API_KEY_HERE" or key == "YOUR_API_KEY_HERE":
        return None
    return key

_api_key = _resolve_key()

if _api_key:
    try:
        genai.configure(api_key=_api_key)
        print("✅ Gemini API configured. Fetching available models...")
        
        models = genai.list_models()
        gemini_models = [m for m in models if 'gemini' in m.name]
        
        if not gemini_models:
            print("❌ No Gemini models found in your account.")
            model = None
        else:
            preferred = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
            selected = None
            for pref in preferred:
                for m in gemini_models:
                    if pref in m.name:
                        selected = m
                        break
                if selected:
                    break
            if not selected:
                selected = gemini_models[0]  # first available
            
            model = genai.GenerativeModel(selected.name.replace('models/', ''))
            print(f"✅ Using Gemini model: {selected.name}")
    except Exception as e:
        model = None
        print(f"❌ Error configuring Gemini: {e}")
else:
    model = None
    print("❌ Gemini not configured (API key is missing or invalid).")

def _call_gemini(prompt, default="Gemini API not available."):
    if not getattr(config, "USE_GEMINI", True) or model is None:
        return default
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def _format_examples(closest_examples):
    if not closest_examples:
        return ""
    lines = ["\nThis patient's vitals closely match the following example(s) from our clinical database:"]
    for i, (dist, ex) in enumerate(closest_examples, 1):
        lines.append(f"  Example {i} (distance {dist:.3f}):")
        lines.append(f"    Vitals: MAP={ex['vitals']['map']}, Lactate={ex['vitals']['lactate']}, "
                     f"HRV={ex['vitals']['hrv']}, ScvO2={ex['vitals']['scvo2']}")
        lines.append(f"    Diagnosis: {ex['diagnosis']}")
        lines.append(f"    Warning: {ex['warning']}")
    return "\n".join(lines)


def generate_clinical_summary(condition, metrics_dict, betti_interpretation, diagnosis, closest_examples=None):
    """
    Generate a concise, clinically relevant summary (3-5 sentences).
    Optionally incorporates closest matching dataset examples.
    """
    examples_text = _format_examples(closest_examples)
    prompt = f"""
You are a clinical decision support system. A patient presents with {condition}. 
Based on topological data analysis of physiological signals, we have the following metrics:

- Betti numbers (β0, β1, β2): {metrics_dict['betti']} 
  Interpretation: {betti_interpretation}
- Maximal Lyapunov exponent: {metrics_dict['lyap_max']:.3f} ({metrics_dict['lyap_interp']})
- Persistent entropy: {metrics_dict['entropy_h1']:.3f}
- Correlation dimension: {metrics_dict['corr_dim']:.3f}
- Approximate entropy: {metrics_dict['apen']:.3f}
- Box-counting dimension: {metrics_dict['box_dim']:.3f}
- Mean Ricci curvature: {metrics_dict['mean_curv']:.3f}
- Most complex variable: {metrics_dict['most_complex_var']} (complexity {metrics_dict['max_complex']:.3f})

The system's diagnosis is: {diagnosis}
{examples_text}

Write a concise, clinically relevant summary (3-5 sentences) explaining what these numbers mean for the patient's condition and why they support the diagnosis. Use plain language that a doctor would understand. If example(s) are provided, you may reference them to add context.
"""
    return _call_gemini(prompt, default="Clinical summary not available.")

def generate_patient_friendly_summary(condition, metrics_dict, diagnosis, closest_examples=None):
    """
    Generate a very simple explanation for the patient or family.
    Optionally incorporates closest matching dataset examples.
    """
    examples_text = _format_examples(closest_examples)
    prompt = f"""
A patient has been diagnosed with {condition}. 
The monitoring system uses advanced mathematics to track the body's state. Here are some key numbers:

- Heart rate variability (HRV) is {metrics_dict.get('hrv_value', 'N/A'):.1f} (normal range 40-100)
- Blood pressure (MAP) is {metrics_dict.get('map_value', 'N/A'):.1f} mmHg
- Inflammation marker IL-6 is {metrics_dict.get('il6_value', 'N/A'):.1f} pg/mL
- The system's overall "health score" is {metrics_dict.get('health_score', 'N/A')} out of 100.

The computer's diagnosis is: {diagnosis}
{examples_text}

Write a short paragraph (2-3 sentences) explaining in very simple terms what is happening and what the doctors are doing. Avoid technical jargon. If example(s) are provided, you may use them to reassure or explain.
"""
    return _call_gemini(prompt, default="Patient-friendly summary not available.")

def generate_treatment_insight(initial_results, treatment_results, closest_examples=None):
    """
    Generate a narrative comparing pre- and post-treatment metrics.
    Optionally incorporates closest matching dataset examples.
    """
    examples_text = _format_examples(closest_examples)
    prompt = f"""
A patient with {treatment_results['condition']} received {treatment_results['treatment']} treatment. 
Before treatment:
- Betti numbers (β0,β1,β2): {initial_results['betti']}
- Lyapunov λ: {initial_results['lyap_max']:.3f}
- Diagnosis: {initial_results['diagnosis']}

After treatment:
- Betti numbers: {treatment_results['betti']}
- Lyapunov λ: {treatment_results['lyap_max']:.3f}
- Diagnosis: {treatment_results['diagnosis']}
{examples_text}

Write a brief paragraph explaining whether the treatment improved the patient's topological state and what the changes in metrics indicate clinically. Also mention if the patient is moving toward a healthier state. If example(s) are provided, you may compare the post-treatment state to similar cases.
"""
    return _call_gemini(prompt, default="Treatment explanation not available.")

def generate_metric_explanation(metric_name, value, interpretation):
    """
    Explain a single metric in simple terms. (No examples needed)
    """
    prompt = f"""
In medical monitoring, we use a mathematical quantity called "{metric_name}" (current value = {value:.3f}). 
Here is a technical interpretation: {interpretation}

Now, please explain this concept in very simple, everyday language that anyone could understand. Use an analogy if helpful.
"""
    return _call_gemini(prompt, default=f"Explanation for {metric_name} not available.")

def generate_synthetic_case_study(condition, metrics_dict, diagnosis, closest_examples=None):
    """
    Create a narrative patient case study based on the data.
    Optionally incorporates closest matching dataset examples.
    """
    examples_text = _format_examples(closest_examples)
    prompt = f"""
Based on the following data from a patient with {condition}, write a short synthetic case study (3-5 sentences) that a medical student could read. Include the presenting complaint, key findings, and the diagnosis.

Data:
- Betti numbers: {metrics_dict['betti']}
- Lyapunov exponent: {metrics_dict['lyap_max']:.3f}
- Most complex variable: {metrics_dict['most_complex_var']}
- Diagnosis: {diagnosis}
{examples_text}

Make it realistic and educational. If example(s) are provided, you may incorporate them as similar cases in the literature.
"""
    return _call_gemini(prompt, default="Synthetic case study not available.")

def generate_health_score_interpretation(health_score, metrics_dict, closest_examples=None):
    """
    Explain the overall health score.
    Optionally incorporates closest matching dataset examples.
    """
    examples_text = _format_examples(closest_examples)
    prompt = f"""
A patient has an overall "health score" of {health_score:.1f} out of 100, calculated from multiple physiological metrics.
The individual metrics are:
- β0 (fragmentation): {metrics_dict['betti'][0]}
- β1 (loops): {metrics_dict['betti'][1]}
- β2 (voids): {metrics_dict['betti'][2]}
- Lyapunov λ: {metrics_dict['lyap_max']:.3f}
- Persistent entropy: {metrics_dict['entropy_h1']:.3f}
{examples_text}

Write a short paragraph interpreting this health score: what it means for the patient's condition and whether it indicates improvement or deterioration. If example(s) are provided, you may compare the score to similar cases.
"""
    return _call_gemini(prompt, default="Health score interpretation not available.")