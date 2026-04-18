import gradio as gr
import pandas as pd
import numpy as np
import joblib

# -------- LOAD --------
best_model = joblib.load('/Workspace/Files/fraud_best_model.pkl')
scaler = best_model.named_steps['scaler']
model = best_model.named_steps['model']
graph_scores = joblib.load('/Workspace/Files/fraud_graph_scores.pkl')

# -------- MAIN FUNCTION --------
def analyze_transaction(
    transaction_type, amount, oldbalanceOrg, newbalanceOrg,
    oldbalanceDest, newbalanceDest, hour,
    sender_id, receiver_id
):

    is_night = int(hour < 6)
    amount_ratio = amount / (oldbalanceOrg + 1)
    orig_balance_zero = int(oldbalanceOrg == 0)
    dest_balance_zero = int(oldbalanceDest == 0)
    type_TRANSFER = int(transaction_type == "TRANSFER")

    X_input = pd.DataFrame([[ 
        1, amount, oldbalanceOrg, newbalanceOrg,
        oldbalanceDest, newbalanceDest, 0,
        hour, is_night, amount_ratio,
        orig_balance_zero, dest_balance_zero, type_TRANSFER
    ]], columns=[
        'step','amount','oldbalanceOrg','newbalanceOrig',
        'oldbalanceDest','newbalanceDest','isFlaggedFraud',
        'hour','is_night','amount_ratio',
        'orig_balance_zero','dest_balance_zero','type_TRANSFER'
    ])

    X_input = X_input.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X_scaled = scaler.transform(X_input)
    ml_score = model.predict_proba(X_scaled)[0,1]

    graph_score = graph_scores.get(sender_id, 0)

    final_score = 0.95*ml_score + 0.05*graph_score

    label = "FRAUD" if final_score > 0.4 else "SAFE"
    color = "#e74c3c" if label=="FRAUD" else "#27ae60"

    return f"""
    <div style="display:flex;gap:20px;">
      <div style="flex:1;">

        <div style="background:#f5f6fa;color:#000000 !important;padding:20px;border-radius:10px;margin-bottom:10px;">
          <b>ML Score:</b> {ml_score:.4f}
        </div>

        <div style="background:#f5f6fa;color:#000000 !important;padding:20px;border-radius:10px;margin-bottom:10px;">
          <b>Graph Score:</b> {graph_score:.4f}
        </div>

        <div style="background:{color};color:white;padding:20px;border-radius:10px;">
          <b>Final Score:</b> {final_score:.4f}<br>{label}
        </div>

      </div>
    </div>
    """

# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("## Fraud Detection System")

    t = gr.Dropdown(["TRANSFER","PAYMENT"])
    amt = gr.Number()
    ob = gr.Number()
    nb = gr.Number()
    obd = gr.Number()
    nbd = gr.Number()
    hr = gr.Slider(0,23)
    sid = gr.Textbox()
    rid = gr.Textbox()

    btn = gr.Button("Analyze")
    out = gr.HTML()

    btn.click(analyze_transaction,
        inputs=[t,amt,ob,nb,obd,nbd,hr,sid,rid],
        outputs=out
    )

demo.launch()
