"""Streamlit demo for the FINAL 19-feature Random Forest pipeline.
Requires the user's own rf_final.joblib and final_config.json generated in Colab.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'rf_final.joblib'
CONFIG_PATH=ROOT/'final_config.json'
MONTHS=['January','February','March','April','May','June','July','August','September','October','November','December']

st.set_page_config(page_title='Hotel Booking Cancellation | SGU',page_icon='🏨',layout='centered')
st.title('🏨 Hotel Booking Cancellation Prediction')
st.caption('Đồ án Máy học – Trường Đại học Sài Gòn | Dự đoán bằng Random Forest 19 feature')

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error('Chưa có rf_final.joblib và final_config.json trong cùng thư mục với app.py. Hãy tải 2 file từ thư mục final_experiment trên Google Drive của nhóm.')
    st.stop()

# Only load model files created and saved by your own team. joblib uses pickle.
model=joblib.load(MODEL_PATH)
config=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
feature_names=config['features']
threshold=float(config.get('threshold',0.32))

if len(feature_names)!=19:
    st.error('Cấu hình hiện tại không có đúng 19 feature. Vui lòng kiểm tra final_config.json.')
    st.stop()

pre=model.named_steps['preprocessor']
cat_cols=list(pre.transformers_[1][2])  # cat branch was the second branch in final training
encoder=pre.named_transformers_['cat'].named_steps['encoder']
options={col:[str(v) for v in cats if pd.notna(v)] for col,cats in zip(cat_cols,encoder.categories_)}

def choose(label,name,fallback=None):
    vals=options.get(name,[]) or ([fallback] if fallback else ['Unknown'])
    idx=0
    if fallback in vals: idx=vals.index(fallback)
    return st.selectbox(label,vals,index=idx)

with st.form('predict'):
    c1,c2=st.columns(2)
    with c1:
        hotel=choose('Loại khách sạn','hotel','City Hotel')
        month=choose('Tháng dự kiến đến','arrival_date_month','July')
        lead=int(st.number_input('Số ngày đặt trước (lead_time)',0,1000,60))
        weekend=int(st.number_input('Số đêm cuối tuần',0,30,1))
        weekday=int(st.number_input('Số đêm trong tuần',0,60,2))
        adults=int(st.number_input('Người lớn',0,20,2))
        children=float(st.number_input('Trẻ em',0,20,0))
        babies=int(st.number_input('Em bé',0,10,0))
        day=int(st.number_input('Ngày nhận phòng trong tháng',1,31,15))
    with c2:
        meal=choose('Gói bữa ăn','meal','BB')
        market=choose('Market segment','market_segment','Online TA')
        channel=choose('Distribution channel','distribution_channel','TA/TO')
        room=choose('Loại phòng đã đặt','reserved_room_type','A')
        customer=choose('Customer type','customer_type','Transient')
        repeat=int(st.selectbox('Khách từng quay lại?',[0,1]))
        prev_cancel=int(st.number_input('Số lần hủy trước',0,100,0))
        prev_not=int(st.number_input('Số booking trước không hủy',0,200,0))
    submit=st.form_submit_button('Dự đoán',type='primary',use_container_width=True)

if submit:
    raw={
        'lead_time':lead,'stays_in_weekend_nights':weekend,'stays_in_week_nights':weekday,
        'adults':adults,'children':children,'babies':babies,'is_repeated_guest':repeat,
        'previous_cancellations':prev_cancel,'previous_bookings_not_canceled':prev_not,
        'arrival_date_day_of_month':day,'total_nights':weekend+weekday,
        'total_guests':adults+children+babies,'hotel':hotel,'arrival_date_month':month,
        'meal':meal,'market_segment':market,'distribution_channel':channel,
        'reserved_room_type':room,'customer_type':customer,
    }
    inp=pd.DataFrame([raw],columns=feature_names)
    prob=float(model.predict_proba(inp)[0,1])
    pred=int(prob>=threshold)
    st.metric('Điểm dự đoán lớp hủy',f'{prob:.1%}')
    if pred:
        st.warning(f'Mô hình phân loại: CÓ NGUY CƠ HỦY (score ≥ {threshold:.2f})')
    else:
        st.success(f'Mô hình phân loại: KHÔNG HỦY (score < {threshold:.2f})')
    st.info('Chỉ dùng để minh họa kỹ thuật. Điểm số chưa được xác nhận hiệu chuẩn thành xác suất thực tế và không nên dùng để tự động ra quyết định đối với khách hàng.')

st.divider()
st.caption('Mô hình final: Random Forest, 19 feature, threshold = 0.32 theo validation; Test ROC-AUC = 0.8329. Chỉ mở file joblib do chính nhóm tạo.')
